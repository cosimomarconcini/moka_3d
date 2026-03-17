#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:39:26 2026

@author: cosimo
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import logging
import shutil
import warnings
import copy

import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18 as cosmo
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
from skimage.transform import downscale_local_mean
from matplotlib.ticker import AutoMinorLocator

from . import moka3d_source as km
from .plotting import finalize_figure
import astropy.units as u

from astropy.io import fits

logger = logging.getLogger(__name__)


            
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",      # cyan
        "INFO": "\033[0m",        # normal
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[1;31m", # bright red
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "moka3d.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter_file = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    formatter_console = ColorFormatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter_console)
    logger.addHandler(sh)


def _save_summary(summary: dict, output_dir: Path) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _disc_zeta_range(cfg):
    if cfg.advanced.disc_zeta_range_mode == "auto_from_psf":
        return [-cfg.processing.psf_sigma / 2.0, cfg.processing.psf_sigma / 2.0]
    raise ValueError(f"Unsupported disc_zeta_range_mode: {cfg.advanced.disc_zeta_range_mode}")

def _ask_user_to_continue_after_mask_check(output_path: Path) -> None:
    print(f"\nMasking preview saved to:\n{output_path}")
    print("Check the masking figure.")
    answer = input("Continue with this masking? [y/n]: ").strip().lower()

    if answer not in {"y", "yes", "Y"}:
        raise RuntimeError(
            "Run stopped by user after masking check. "
            "Edit the YAML file and run again."
        )


def _plot_mask_preview(
    obs,
    mask_cone_pos,
    mask_cone_neg,
    mask_bicone,
    output_dir: Path,
    show_plots: bool,
    agn_xy_pix,
    arcsec_per_pix,
    xrange,
    yrange,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    flux = np.array(obs.maps["flux"], copy=True)
    flux_plot = np.log10(flux)

    ny, nx = flux.shape
    x0, y0 = agn_xy_pix

    # pixel-center coordinates in arcsec, with AGN at (0,0)
    x_arc = (np.arange(nx) - x0) * arcsec_per_pix
    y_arc = (np.arange(ny) - y0) * arcsec_per_pix

    # extent for imshow
    extent = [
        x_arc[0] - 0.5 * arcsec_per_pix,
        x_arc[-1] + 0.5 * arcsec_per_pix,
        y_arc[0] - 0.5 * arcsec_per_pix,
        y_arc[-1] + 0.5 * arcsec_per_pix,
    ]

    X_arc, Y_arc = np.meshgrid(x_arc, y_arc)

    panels = [
        ("Outflow mask (+)", mask_cone_pos),
        ("Outflow mask (-)", mask_cone_neg),
        ("Outflow mask (bicone)", mask_bicone),
    ]

    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(flux_plot, origin="lower", extent=extent, cmap = 'inferno')
        ax.contour(X_arc, Y_arc, mask.astype(float), levels=[0.5], linewidths=2)

        # AGN at (0,0)
        ax.plot(
            0.0, 0.0,
            marker="*",
            markersize=16,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            zorder=10,
        )

        ax.set_title(title)
        ax.set_xlabel(r'$\Delta$ RA ["]')
        ax.set_ylabel(r'$\Delta$ Dec ["]')

        if xrange is not None:
            ax.set_xlim(xrange)
        if yrange is not None:
            ax.set_ylim(yrange)

    plt.tight_layout()
    outpath = output_dir / "02a_mask_preview.png"
    finalize_figure(outpath, show=show_plots)

    return outpath




def _resolve_display_range(data, cfg_range, positive_mask=None):
    arr = np.asarray(data, dtype=float)

    if positive_mask is not None:
        arr = arr[positive_mask]
    else:
        arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return [np.nan, np.nan]

    mode = str(cfg_range.mode).lower()
    values = list(cfg_range.values)

    if len(values) != 2:
        raise ValueError("Display range 'values' must contain exactly two numbers.")

    if mode == "percentile":
        return list(np.nanpercentile(arr, values))

    if mode == "fixed":
        return [float(values[0]), float(values[1])]

    raise ValueError(f"Unsupported display range mode: {cfg_range.mode}")





def _plot_escape_fraction_profile(
    *,
    output_path: Path,
    scale_kpc_per_arcsec: float,
    radius_range_model_disc,
    radius_range_model_out,
    disc_profile,
    out_pos_profile=None,
    out_neg_profile=None,
    out_avg_profile=None,
    show_plots=False,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=3)
    ax.tick_params(axis="both", labelsize=12)

    def _draw_one(profile, label):
        if profile is None:
            return

        r = np.asarray(profile["r_arcsec"], dtype=float)
        xerr = np.asarray(profile["xerr_arcsec"], dtype=float)
        y = np.asarray(profile["ratio"], dtype=float)
        yerr = np.asarray(profile["ratio_err"], dtype=float)

        good = np.isfinite(r) & np.isfinite(y)
        if not np.any(good):
            return

        # --- systematic halo-size band: eta=10 to eta=100
        if ("ratio_loweta" in profile) and ("ratio_higheta" in profile):
            y_eta_low = np.asarray(profile["ratio_loweta"], dtype=float)
            y_eta_high = np.asarray(profile["ratio_higheta"], dtype=float)

            y_sys_lo = np.minimum(y_eta_low, y_eta_high)
            y_sys_hi = np.maximum(y_eta_low, y_eta_high)

            ok_sys = good & np.isfinite(y_sys_lo) & np.isfinite(y_sys_hi)
            if np.any(ok_sys):
                ax.fill_between(
                    r[ok_sys],
                    y_sys_lo[ok_sys],
                    y_sys_hi[ok_sys],
                    alpha=0.15,
                    #label=f"{label} halo range",
                )

        # --- statistical uncertainty band around fiducial eta=30
        #ylo = y - yerr
        #yhi = y + yerr
        #ok_stat = good & np.isfinite(ylo) & np.isfinite(yhi)
        #if np.any(ok_stat):
        #    ax.fill_between(r[ok_stat], ylo[ok_stat], yhi[ok_stat], alpha=0.25)

        # --- fiducial eta=30 points/line
        ax.errorbar(
            r[good], y[good],
            xerr=xerr[good],
            #yerr=yerr[good],
            fmt="o-",
            lw=1.5,
            capsize=4,
            label=label,
        )

    _draw_one(out_pos_profile, "Outflow (+)")
    _draw_one(out_neg_profile, "Outflow (-)")
    _draw_one(out_avg_profile, "Outflow avg")

    ax.axhline(1.0, ls="--", lw=1.2, color = 'black')
    ax.set_xlabel(r"Radius [arcsec]", fontsize=14)
    ax.set_ylabel(r"$v_{\rm out}/(v_{\rm esc})$", fontsize=14)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=11, loc="best")

    rmax_arc = float(max(radius_range_model_disc[1], radius_range_model_out[1]))
    xmin, xmax = 0.0, rmax_arc
    pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.1
    ax.set_xlim(max(0.0, xmin), xmax + pad)

    def a2k(x):
        return x * float(scale_kpc_per_arcsec)

    def k2a(x):
        return x / float(scale_kpc_per_arcsec)

    secax = ax.secondary_xaxis("top", functions=(a2k, k2a))
    secax.set_xlabel("Radius [kpc]", fontsize=14)
    secax.tick_params(axis="both", labelsize=12)


    plt.tight_layout()
    finalize_figure(output_path, show=show_plots)



def _plot_outflow_energetics_profile(
    *,
    output_path: Path,
    scale_kpc_per_arcsec: float,
    radius_range_model_out,
    pos_profile=None,
    neg_profile=None,
    show_plots=False,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=3)
    ax.tick_params(axis="both", labelsize=12)

    def _draw(profile, label):
        if profile is None:
            return

        r = np.asarray(profile["r_arcsec"], dtype=float)
        xerr = 0.5 * np.asarray(profile["dr_arcsec"], dtype=float)

        # standard-density fallback: draw shaded band
        if "mdot_lo_msun_yr" in profile and "mdot_hi_msun_yr" in profile:
            lo = np.asarray(profile["mdot_lo_msun_yr"], dtype=float)
            hi = np.asarray(profile["mdot_hi_msun_yr"], dtype=float)
            mid = np.asarray(profile["mdot_mid_msun_yr"], dtype=float)

            good = np.isfinite(r) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(mid) & (lo > 0) & (hi > 0) & (mid > 0)
            if np.any(good):
                ax.fill_between(r[good], lo[good], hi[good], alpha=0.2)
                ax.errorbar(r[good], mid[good], xerr=xerr[good], fmt="o-", lw=1.5, capsize=4, label=label)
        else:
            y = np.asarray(profile["mdot_msun_yr"], dtype=float)
            good = np.isfinite(r) & np.isfinite(y) & (y > 0)
            if np.any(good):
                ax.errorbar(r[good], y[good], xerr=xerr[good], fmt="o-", lw=1.5, capsize=4, label=label)

    _draw(pos_profile, "Outflow (+)")
    _draw(neg_profile, "Outflow (-)")

    ax.set_xlabel(r"Radius [arcsec]", fontsize=14)
    ax.set_ylabel(r"$\dot{M}_{\rm out}$ [$M_\odot$ yr$^{-1}$]", fontsize=14)
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=11, loc="best")

    xmin, xmax = 0.0, float(radius_range_model_out[1])
    pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.1
    ax.set_xlim(xmin, xmax + pad)

    def a2k(x):
        return x * float(scale_kpc_per_arcsec)

    def k2a(x):
        return x / float(scale_kpc_per_arcsec)

    secax = ax.secondary_xaxis("top", functions=(a2k, k2a))
    secax.set_xlabel("Radius [kpc]", fontsize=14)
    secax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    finalize_figure(output_path, show=show_plots)




def run_pipeline(cfg, config_path: Path | None = None) -> dict:
    warnings.filterwarnings("ignore", category=VerifyWarning)
    warnings.filterwarnings("ignore", category=FITSFixedWarning)

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 18

    run_name = Path(cfg.input.cube_file).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = cfg.paths.output_dir / f"{timestamp}_{run_name}"
    _setup_logging(output_dir)

    if config_path is not None and cfg.output.save_run_config_copy:
        shutil.copy2(config_path, output_dir / "used_config.yaml")

    logger.info("Starting MOKA3D run")
    logger.info("Cube file: %s", cfg.input.cube_file)

    obscube, obshead, wcs2d, wcs_large, hdu_index = km.load_cube_and_wcs(
        str(cfg.paths.data_dir), cfg.input.cube_file
    )

    bunit_header = obshead.get("BUNIT", None)
    flux_unit_scale = km.flux_unit_scale_from_bunit(bunit_header)

    if bunit_header is not None:
        logger.info("Input cube BUNIT = %s", bunit_header)

    if flux_unit_scale is None:
        logger.warning(
            "Could not parse BUNIT from the input FITS header. "
            "Energetics will assume the cube is already in erg s^-1 cm^-2 Angstrom^-1. "
            "This may cause an order-of-magnitude unit error in the energetics calculation."
        )
        flux_unit_scale = 1.0
    else:
        logger.info(
            "Energetics flux-unit scale factor derived from BUNIT = %.6e",
            flux_unit_scale,
        )


    if cfg.input.sn_map is not None:
        obscube = km.apply_sn_mask_to_cube(
            obscube,
            str(cfg.paths.ancillary_dir),
            cfg.input.sn_map,
            cfg.processing.sn_thresh
        )
        logger.info("SN mask applied: (threshold=%.0f)", cfg.processing.sn_thresh)
    else:
        logger.info("SN masking skipped (sn_thresh=None)")

    pixscale = km.pixel_scale_arcsec(wcs_large)
    pixscale_manual = getattr(cfg.processing, "pixel_scale_arcsec_manual", None)

    if not np.isfinite(pixscale) or pixscale <= 0:
        if pixscale_manual is not None and np.isfinite(float(pixscale_manual)) and float(pixscale_manual) > 0:
            pixscale = float(pixscale_manual)
            logger.warning(
                "Pixel scale could not be inferred from WCS/header. "
                "Using manual value from configuration .yaml file: %.4f arcsec/pix",
               pixscale,
            )
        else:
            hdr_msg = km._describe_missing_pixscale_header_info(obshead)
            raise RuntimeError(
                hdr_msg
                + "\nSet processing.pixel_scale_arcsec_manual in the YAML file "
                  "(for example 0.2 if that is the correct arcsec/pixel value for your data)."
            )



    n_spec = int(obshead.get("NAXIS3", 0))
    spec_coord, spec_unit = km.spectral_axis_from_header_general(obshead, n_spec)

    vel_kms_approx, spec_kind, line_obs = km.velocity_axis_from_spectral_coord(
        spec_coord,
        spec_unit,
        line_value=cfg.line.wavelength_line,
        line_unit=cfg.line.wavelength_line_unit,
        redshift=cfg.target.redshift,
        convention="optical",
    )

    obscube, _ = km.standardize_cube_to_spec_yx(obscube, n_spec=len(vel_kms_approx))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        floor_map = np.nanmin(obscube, axis=0)

    #floor_map[~np.isfinite(floor_map)] = 0.0

    #obscube = obscube - floor_map[None, :, :]
    #obscube = np.clip(obscube, 0.0, None)

    i0 = int(np.nanargmin(np.abs(vel_kms_approx)))
    vel_kms = np.array(vel_kms_approx, copy=True)

    x_cen, y_cen, sx, sy, center_used = km.pick_center(
        wcs2d,
        obscube,
        vel_kms,
        cfg.target.agn_ra,
        cfg.target.agn_dec,
        cfg.target.center_mode,
        tuple(cfg.target.center_xy_manual) if cfg.target.center_xy_manual else None,
        flux_kwargs=dict(vwin_kms=300.0, box=7),
        kin_kwargs=dict(vwin_kms=500.0, box=7, flux_q=70),
    )

    Dc, D_L, D_A, tL, scale = km.distances_from_z(cfg.target.redshift, cosmo)

   
    crpix, crval, cdelt, dv, ref_pix_0based, x0_bin, y0_bin = km.observed_wcs_params_from_vel(
        vel_kms, i0, x_cen, y_cen, pixscale, cfg.processing.nrebin
    )

    if cfg.processing.nrebin > 1:
        obs = km.observed(
            downscale_local_mean(obscube, (1, cfg.processing.nrebin, cfg.processing.nrebin)),
            error=None,
            crval=crval,
            cdelt=cdelt,
            crpix=crpix,
        )
    else:
        obs = km.observed(
            obscube,
            error=None,
            crval=crval,
            cdelt=cdelt,
            crpix=crpix,
            fluxmap=cfg.maps.fluxmap,
            velmap=cfg.maps.velmap,
            sigmap=cfg.maps.sigmap,
        )

    origin = np.array([int(np.round(x0_bin)), int(np.round(y0_bin))], dtype=int)
    xy_AGN = [0.0, 0.0]

    pos = np.isfinite(obs.maps["flux"]) & (obs.maps["flux"] > 0)


    flux_pos_mask = np.isfinite(obs.maps["flux"]) & (obs.maps["flux"] > 0)

    flrange = _resolve_display_range(
        obs.maps["flux"],
        cfg.processing.display_ranges["flux"],
        positive_mask=flux_pos_mask,
    )

    velrange = _resolve_display_range(
        obs.maps["vel"],
        cfg.processing.display_ranges["vel"],
    )

    sigrange = _resolve_display_range(
        obs.maps["sig"],
        cfg.processing.display_ranges["sig"],
    )



    obs.plot_kin_maps(
        flrange=flrange,
        vrange=velrange,
        sigrange=sigrange,
        xy_AGN=xy_AGN,
        xrange=cfg.processing.xrange,
        yrange=cfg.processing.yrange,
    )
    finalize_figure(output_dir / "01_observed_kin_maps.png", show=cfg.output.show_plots)

    R_obs_arcsec, R_int_arcsec, R_int_err, boot_R = km.estimate_radius_from_encircled_flux_with_uncertainty(
        flux_map=obs.maps["flux"],
        center_xy_pix=origin,
        pixscale_arcsec=pixscale,
        nrebin=cfg.processing.nrebin,
        psf_sigma_arcsec=cfg.processing.psf_sigma,
    )

    pa_est, pa_est_unc = km.estimate_pa_from_mom1(
        obs.maps["vel"],
        center_xy=origin,
        pixscale=pixscale,
        nrebin=cfg.processing.nrebin,
        xlimshow=cfg.processing.xrange,
        ylimshow=cfg.processing.yrange,
        psf_sigma_arcsec=cfg.processing.psf_sigma,
        R_data_arcsec=R_int_arcsec,
        R_data_err_arcsec=R_int_err,
        vel_range = velrange
    )
    finalize_figure(output_dir / "03_PA_estimate.png", show=cfg.output.show_plots)
    if (cfg.fit.component_mode == "disk") and bool(cfg.advanced.check_masking_before_fitting):
        print(f"\nPA estimate preview saved to:\n{output_dir / '03_PA_estimate.png'}")
        print("Check the PA estimate figure.")
        answer = input("Continue with this PA estimate? [y/n]: ").strip().lower()

        if answer not in {"y", "yes", "Y"}:
            raise RuntimeError(
                "Run stopped by user after PA check. "
                "Edit the YAML file and run again."
            )


    summary = {
        "input_cube": cfg.input.cube_file,
        "hdu_index": hdu_index,
        "spec_unit": str(spec_unit),
        "pixel_scale_arcsec": float(pixscale),
        "luminosity_distance_mpc": float(D_L.to_value("Mpc")),
        "angular_diameter_distance_mpc": float(D_A.to_value("Mpc")),
        "scale_kpc_per_arcsec": float(scale),
        "estimated_pa_deg": float(pa_est),
        "estimated_pa_unc_deg": float(pa_est_unc),
        "estimated_radius_arcsec": float(R_int_arcsec),
        "estimated_radius_unc_arcsec": float(R_int_err),
        "center_method": center_used,
        "x_center_pix": float(x_cen),
        "y_center_pix": float(y_cen),
    }

    if cfg.output.save_summary_json:
        _save_summary(summary, output_dir)

    logger.info("Initial setup complete")
    logger.info("Estimated PA with no mask = %.2f +/- %.2f deg", pa_est, pa_est_unc)
    
    

    # ============================================================
    # FIT CONFIGURATION FROM YAML FILE
    # ============================================================
    
    FIT_COMPONENT_MODE = cfg.fit.component_mode

    SAVE_ALL_OUTPUTS = bool(cfg.input.save_all_outputs)
    final_model = None
    final_model_name = None


    
    USE_CRPS = cfg.advanced.use_crps
    loss = "crps" if USE_CRPS else "extreme"
    
    perc_disc = list(cfg.advanced.perc_disc)
    perc_out = list(cfg.advanced.perc_out)
    perc_weights = list(cfg.advanced.perc_weights)
    
    npt = int(cfg.advanced.npt)
    
    disc_cfg = cfg.fit.disc
    out_cfg = cfg.fit.outflow

    radius_range_model_disc = list(disc_cfg.radius_range_arcsec)
    radius_range_model_out = list(out_cfg.radius_range_arcsec)

    num_shells_disc = int(disc_cfg.num_shells)
    num_shells_out = int(out_cfg.num_shells)

    beta_min_d, beta_max_d, step_beta_d = map(float, disc_cfg.beta_grid_deg)
    beta_min_o, beta_max_o, step_beta_o = map(float, out_cfg.beta_grid_deg)

    OUTFLOW_PA_DEG = float(out_cfg.pa_deg)
    OUTFLOW_OPENING_DEG = float(out_cfg.opening_deg)
    OUTFLOW_DOUBLE_CONE = bool(out_cfg.double_cone)
    OUTFLOW_MASK_MODE = str(out_cfg.mask_mode)
    OUTFLOW_AXIS_SIGN = int(cfg.advanced.outflow_axis_sign)

    v_min_o, v_max_o, step_v_o = map(float, out_cfg.v_grid_kms)

    
    USE_GLOBAL_BETA_DISC = bool(cfg.advanced.use_global_beta_disc)
    DISC_FIT_MODE = str(cfg.fit.disc.mode)
    DISC_PA_DEG = None if disc_cfg.pa_deg is None else float(disc_cfg.pa_deg)
    disc_phi_range = cfg.advanced.disc_phi_range
    disc_zeta_range = _disc_zeta_range(cfg)

    DISC_IS_PHYSICAL = DISC_FIT_MODE in {"disk_kepler", "NSC", "Plummer", "disk_arctan"}

    disc_mode_cfg = disc_cfg

    DISC_N_GEOM_V = 50
    DISC_R_NSC_PC = None
    DISC_A_PLU_PC = None
    DISC_RT_ARCSEC = None

    if DISC_FIT_MODE == "independent":
        v_min_d, v_max_d, step_v_d = map(float, disc_mode_cfg.independent.v_grid_kms)

    elif DISC_FIT_MODE == "disk_kepler":
        v_min_d, v_max_d = map(float, disc_mode_cfg.kepler.mbh_grid_msun)
        step_v_d = 1.0
        DISC_N_GEOM_V = int(disc_mode_cfg.kepler.n_geom)

    elif DISC_FIT_MODE == "NSC":
        v_min_d, v_max_d = map(float, disc_mode_cfg.nsc.a_grid)
        step_v_d = 1.0
        DISC_N_GEOM_V = int(disc_mode_cfg.nsc.n_geom)
        DISC_R_NSC_PC = float(disc_mode_cfg.nsc.re_pc)

    elif DISC_FIT_MODE == "Plummer":
        v_min_d, v_max_d = map(float, disc_mode_cfg.plummer.m0_grid_msun)
        step_v_d = 1.0
        DISC_N_GEOM_V = int(disc_mode_cfg.plummer.n_geom)
        DISC_A_PLU_PC = float(disc_mode_cfg.plummer.a_pc)

    elif DISC_FIT_MODE == "disk_arctan":
        v_min_d, v_max_d, step_v_d = map(float, disc_mode_cfg.arctan.vmax_grid_kms)
        DISC_RT_ARCSEC = None if disc_mode_cfg.arctan.rt_arcsec is None else float(disc_mode_cfg.arctan.rt_arcsec)

    else:
        raise ValueError(f"Unsupported DISC_FIT_MODE: {DISC_FIT_MODE}")


    if DISC_IS_PHYSICAL and not USE_GLOBAL_BETA_DISC:
        logger.warning(
            "DISC physical mode '%s' is not compatible with per-shell free-v summary. "
            "Forcing USE_GLOBAL_BETA_DISC=True.",
            DISC_FIT_MODE
        )
        USE_GLOBAL_BETA_DISC = True

    if DISC_FIT_MODE == "NSC" and DISC_R_NSC_PC is None:
        raise ValueError("For DISC_FIT_MODE='NSC' you must set fit.disc.nsc.re_pc.")

    if DISC_FIT_MODE == "Plummer" and DISC_A_PLU_PC is None:
        raise ValueError("For DISC_FIT_MODE='Plummer' you must set fit.disc.plummer.a_pc.")

    if DISC_FIT_MODE == "disk_arctan" and DISC_RT_ARCSEC is None:
        raise ValueError("For DISC_FIT_MODE='disk_arctan' you must set fit.disc.arctan.rt_arcsec.")


    
    USE_GLOBAL_BETA_OUT = bool(cfg.advanced.use_global_beta_out)
    
    MASK_DISK_WITH_OUTFLOW = bool(cfg.advanced.mask_disk_with_outflow)
    # MASK_MODE = str(cfg.advanced.mask_mode)
    DO_FINAL_COMBINED_MODEL_PLOT = bool(cfg.advanced.do_final_combined_model_plot)
    resid_ranges = list(cfg.advanced.resid_ranges)

    COMPUTE_ESCAPE_FRACTION = bool(cfg.advanced.compute_escape_fraction)
    SAVE_ESCAPE_FRACTION_TABLE = bool(cfg.advanced.save_escape_fraction_table)

    ESCAPE_ETA_LOW = 10.0
    ESCAPE_ETA_FID = 30.0
    ESCAPE_ETA_HIGH = 100.0

    COMPUTE_ENERGETICS = bool(cfg.advanced.compute_energetics)
    SAVE_ENERGETICS_TABLE = bool(cfg.advanced.save_energetics_table)

    NE_MAP_NAME = cfg.input.ne_map
    NE_OUTFLOW = cfg.input.ne_outflow
    ASSUMED_NE_VALUES = list(cfg.advanced.assumed_ne_values)
    OIII_METALLICITY_Z_OVER_ZSUN = float(cfg.advanced.oiii_metallicity_z_over_zsun)
    ne_map_2d = None
    if NE_MAP_NAME is not None:
        ne_map_path = Path(cfg.paths.ancillary_dir) / str(NE_MAP_NAME)
        ne_map_2d = km.load_ne_map(ne_map_path)
        logger.info("Density map loaded for energetics: %s", ne_map_path)


    
    # Processing/runtime aliases used throughout the old script
    vel = vel_kms
    nrebin = int(cfg.processing.nrebin)
    psf_sigma = float(cfg.processing.psf_sigma)
    lsf_sigma = float(cfg.processing.lsf_sigma)
    vel_sigma = float(cfg.processing.vel_sigma)
    xrange = cfg.processing.xrange
    yrange = cfg.processing.yrange
        
    dv_chan = float(np.nanmedian(np.abs(np.diff(vel))))
    SIGMA_PERC_KMS = float(np.sqrt(lsf_sigma**2 + (0.5 * dv_chan)**2))
    logradius = False
    
    arcsec_per_pix = pixscale * nrebin
    rin_pix_disc = int(round(radius_range_model_disc[0] / arcsec_per_pix))
    rout_pix_disc = int(round(radius_range_model_disc[1] / arcsec_per_pix))
    rin_pix_out = int(round(radius_range_model_out[0] / arcsec_per_pix))
    rout_pix_out = int(round(radius_range_model_out[1] / arcsec_per_pix))
    
    cube_range = obs.cube["range"]
    cube_nbins = obs.cube["nbins"]
    
    logger.info("Component mode: %s | fit mode: %s",FIT_COMPONENT_MODE,DISC_FIT_MODE)
    if FIT_COMPONENT_MODE == "disk":
        logger.info("Disc shells: %d", num_shells_disc)

    elif FIT_COMPONENT_MODE == "outflow":
        logger.info("Outflow shells: %d", num_shells_out)

    elif FIT_COMPONENT_MODE == "disk_then_outflow":
        logger.info(
            "Disc shells: %d | Outflow shells: %d",
            num_shells_disc,
            num_shells_out,
        )
    
    # fixed standard params that could be also decided by the user even if are std
    CRPS_QGRID = np.linspace(0.01, 0.99, 19)   
    disc_geometry = "cylindrical" # disc geometry is always cylindrical
    disc_theta_range = list(cfg.advanced.disc_theta_range)
    disc_double_cone = False
    disc_aperture = disc_theta_range[0][1] * 2
    # --------------------------
    # OUTFLOW component
    # --------------------------
    OUTFLOW_FIT_MODE = "independent" # outflow mode is always independent
    out_geometry = "spherical" # outflow geometry is always spherical
    # --------------------------
    # Combined mode: mask outflow region for disc fit
    # --------------------------
    MASK_MODE = "zero"  # "zero" recommended
    
    # --------------------------
    # --- lobe + (axis_sign = +1)
    # --------------------------
    # Build outflow spatial masks only when outflow is part of the fit
    # --------------------------
    mask_cone_pos = None
    mask_cone_neg = None
    mask_bicone = None

    if FIT_COMPONENT_MODE in ("outflow", "disk_then_outflow"):
        mask_cone_pos = km.make_cone_spatial_mask(
            shape_yx=obs.cube["data"].shape[1:],
            center_xy=origin,
            pa_deg=OUTFLOW_PA_DEG,
            opening_deg=OUTFLOW_OPENING_DEG,
            mode="single",
            axis_sign=+1
        )

        mask_cone_neg = km.make_cone_spatial_mask(
            shape_yx=obs.cube["data"].shape[1:],
            center_xy=origin,
            pa_deg=OUTFLOW_PA_DEG,
            opening_deg=OUTFLOW_OPENING_DEG,
            mode="single",
            axis_sign=-1
        )

        mask_bicone = (mask_cone_pos | mask_cone_neg)

        if bool(cfg.advanced.check_masking_before_fitting):
            mask_preview_path = _plot_mask_preview(
                obs=obs,
                mask_cone_pos=mask_cone_pos,
                mask_cone_neg=mask_cone_neg,
                mask_bicone=mask_bicone,
                output_dir=output_dir,
                show_plots=cfg.output.show_plots,
                agn_xy_pix =origin,
                arcsec_per_pix = arcsec_per_pix,
                xrange=xrange,
                yrange=yrange,
            )


            
            if DISC_PA_DEG is not None:
                logger.info(
                        "check_masking_before_fitting=True but disc_pa_deg is fixed: "
                        "(%.0f deg), so no interactive confirmation is requested.",
                        float(DISC_PA_DEG),
                    )
            else:
                _ask_user_to_continue_after_mask_check(mask_preview_path)

    def _keep_only_mask(cube, keep_mask_yx, mode="nan"):
        """Keep pixels inside keep_mask_yx; mask everything else."""
        mask_outside = ~keep_mask_yx
        return km.apply_spatial_mask_to_cube(cube, mask_outside, mode=mode)


    def _validate_ne_outflow_list(ne_list, fit_bicone, axis_sign):
        if ne_list is None:
            return

        if not isinstance(ne_list, (list, tuple)):
            raise ValueError("input.ne_outflow must be null or a list.")

        if fit_bicone:
            if len(ne_list) != 2:
                raise ValueError(
                    "For bicone outflow, input.ne_outflow must contain two values: [ne_plus, ne_minus]."
                )
        else:
            if len(ne_list) != 1:
                raise ValueError(
                    "For single-cone outflow, input.ne_outflow must contain one value: [ne_single]."
                )


    fit_bicone = (str(OUTFLOW_MASK_MODE).lower() == "bicone") or bool(OUTFLOW_DOUBLE_CONE)
    if FIT_COMPONENT_MODE in ("outflow", "disk_then_outflow"):
        _validate_ne_outflow_list(NE_OUTFLOW, fit_bicone, OUTFLOW_AXIS_SIGN)


    # ========================================
    # 2) DISC FIT
    # ========================================
    disc_fit = None
    model_disc_best = None
    obs_disc_fit = None
    disc_best2_for_plots = None
    disc_cube_for_outflow = None
    gamma_disc = None
    gamma_disc_unc = None

    if FIT_COMPONENT_MODE in ("disk", "disk_then_outflow"):

        cube_disc_fit = obs.cube["data"]

        if (FIT_COMPONENT_MODE == "disk_then_outflow") and MASK_DISK_WITH_OUTFLOW:
            # mask union of both lobes if bicone requested; otherwise mask the single cone region
            if str(OUTFLOW_MASK_MODE).lower() == "bicone" or bool(OUTFLOW_DOUBLE_CONE):
                mask_for_disc = mask_bicone
            else:
                mask_for_disc = mask_cone_pos if (OUTFLOW_AXIS_SIGN >= 0) else mask_cone_neg

            cube_disc_fit = km.apply_spatial_mask_to_cube(obs.cube["data"], mask_for_disc, mode=MASK_MODE)
        obs_disc_fit = km.make_observed_like(obs, cube_disc_fit)
        obs_disc_fit.plot_kin_maps(flrange=flrange, vrange=velrange, sigrange=sigrange,
                                   xy_AGN=xy_AGN, xrange=xrange, yrange=yrange)
        finalize_figure(output_dir / "02_disc_fit_input_maps.png", show=cfg.output.show_plots)
        if disc_cfg.pa_deg is None:
            gamma_disc, gamma_disc_unc = km.estimate_pa_from_mom1(
                obs_disc_fit.maps["vel"],
                center_xy=origin,
                pixscale=pixscale,
                nrebin=nrebin,
                xlimshow=xrange,
                ylimshow=yrange,
                psf_sigma_arcsec=psf_sigma,
                R_data_arcsec=R_int_arcsec,
                R_data_err_arcsec=R_int_err,
                vel_range = velrange
            )
            logger.info(
                "DISC PA estimated from moment-1 map: %.1f +/- %.1f deg",
                gamma_disc,
                gamma_disc_unc,
            )
        else:
            gamma_disc = float(disc_cfg.pa_deg)
            gamma_disc_unc = (
                float(disc_cfg.pa_unc_deg)
                if disc_cfg.pa_unc_deg is not None
                else 0.0
                 )
            logger.info(
                "DISC PA fixed from YAML: %.1f deg (uncertainty=%.1f deg)",
                gamma_disc,
                gamma_disc_unc,
            )
        logger.info("DISC PA adopted: %.1f +/- %.1f deg", gamma_disc, gamma_disc_unc)
        # disc fit must NOT include any disc_cube
        km.set_fit_context(disc_cube=None)

        disc_fit = km.fit_gridsearch_component(
            obs_for_fit=obs_disc_fit,
            vel_axis=vel,
            origin=origin,
            pixscale=pixscale,
            nrebin=nrebin,
            scale=scale,
            geometry=disc_geometry,
            FIT_MODE=DISC_FIT_MODE,
            gamma_model_deg=gamma_disc,
            aperture_deg=disc_aperture,
            double_cone=disc_double_cone,
            radius_range_model_arcsec=radius_range_model_disc,
            theta_range=disc_theta_range,
            phi_range=disc_phi_range,
            zeta_range=disc_zeta_range,
            logradius=logradius,
            psf_sigma=psf_sigma,
            lsf_sigma=lsf_sigma,
            vel_sigma=vel_sigma,
            npt=npt,
            num_shells=num_shells_disc,
            perc=perc_disc,
            perc_weights=perc_weights,
            loss=loss,
            CRPS_QGRID=CRPS_QGRID,
            SIGMA_PERC_KMS=SIGMA_PERC_KMS,
            beta_min=beta_min_d, beta_max=beta_max_d, step_beta=step_beta_d,
            v_min=v_min_d, v_max=v_max_d, step_v=step_v_d,
            R_nsc=DISC_R_NSC_PC,
            a_plu=DISC_A_PLU_PC,
            RT_ARCSEC=DISC_RT_ARCSEC,
            n_geom_v=DISC_N_GEOM_V,
            verbose_label="DISC"
        )

        disc_best_info = km._extract_best_fit_with_uncertainties(disc_fit)

        disc_param_label = {
            "disk_kepler": "M_BH",
            "NSC": "A",
            "Plummer": "M0",
            "disk_arctan": "Vmax",
        }.get(DISC_FIT_MODE, "v")

        disc_param_unit = {
            "disk_kepler": "Msun",
            "NSC": "",
            "Plummer": "Msun",
            "disk_arctan": "km/s",
        }.get(DISC_FIT_MODE, "km/s")


        if disc_param_unit:
            logger.info(
                "DISC Global best: beta=%.1f ± %.1f deg, %s=%.4g ± %.4g %s",
                disc_best_info["beta_best"],
                disc_best_info["beta_err"],
                disc_param_label,
                disc_best_info["v_best"],
                disc_best_info["v_err"],
                disc_param_unit,
            )
        else:
            logger.info(
                "DISC Global best: beta=%.1f ± %.1f deg, %s=%.4g ± %.4g",
                disc_best_info["beta_best"],
                disc_best_info["beta_err"],
                disc_param_label,
                disc_best_info["v_best"],
                disc_best_info["v_err"],
            )






        num_shells_disc_eff = int(np.shape(disc_fit["chi_squared_map"])[0])


        # global chi2(beta)
        beta_best_global_disc, _, _, _ = km.plot_chi2_vs_beta_global(
            disc_fit["chi_squared_map"],
            disc_fit["beta_array"],
            disc_fit["v_array"],
            USE_GLOBAL_BETA=USE_GLOBAL_BETA_DISC,
            reduce_v="min",
            combine_shells="sum",
            logy=True,
            title=r"DISC: global $\chi^2$ vs $\beta$"
        )
        finalize_figure(output_dir / "07_disc_global_beta_chi2.png", show=cfg.output.show_plots)

        if DISC_FIT_MODE in {"disk_kepler", "NSC", "Plummer", "disk_arctan"}:
            bestp = disc_fit.get("best", None)

            param_label = {
                "disk_kepler": r"$M_\bullet$ ($M_\odot$)",
                "NSC": r"$A$",
                "Plummer": r"$M_0$ ($M_\odot$)",
                "disk_arctan": r"$V_{\max}$ (km s$^{-1}$)",
            }[DISC_FIT_MODE]

            use_logx = DISC_FIT_MODE in {"disk_kepler", "NSC", "Plummer"}

            fig = km.plot_chi2_vs_param_global(
                bestp,
                disc_fit["v_array"],
                title=f"DISC: global $\\chi^2$ vs {param_label}",
                x_label=param_label,
                logx=use_logx,
                logy=True,
            )
            if fig is not None:
                finalize_figure(output_dir / "07b_disc_global_param_chi2.png", show=cfg.output.show_plots)

        if USE_GLOBAL_BETA_DISC:
            disc_fit["beta_best"] = float(beta_best_global_disc)

        # summarize for plots + scatter
        disc_best2_for_plots = None

        try:
            if DISC_FIT_MODE in {"disk_kepler", "NSC", "Plummer", "disk_arctan"}:
                # physical/global disc model: build velocity profile from the best global parameter
                best = disc_fit["best"]

                if DISC_FIT_MODE == "disk_kepler":
                    p_best = float(best["p_star"])
                    p_err  = float(best["p_err"])
                elif DISC_FIT_MODE == "NSC":
                    p_best = float(best["p_star"])
                    p_err  = float(best["p_err"])
                elif DISC_FIT_MODE == "Plummer":
                    p_best = float(best["p_star"])
                    p_err  = float(best["p_err"])
                elif DISC_FIT_MODE == "disk_arctan":
                    p_best = float(best["p_star"])
                    p_err  = float(best["p_err"])

                if DISC_FIT_MODE == "disk_arctan":
                    r_arcsec, xerr_arcsec = km._shell_midpoints_and_halfwidths_arcsec(
                        rin_pix=rin_pix_disc,
                        rout_pix=rout_pix_disc,
                        n_shells=num_shells_disc_eff,
                        arcsec_per_pix=arcsec_per_pix,
                    )
                    v = km.vrot_arctan(r_arcsec, None, None, [p_best, DISC_RT_ARCSEC])
                    if np.isfinite(p_err):
                        v_hi = km.vrot_arctan(r_arcsec, None, None, [p_best + p_err, DISC_RT_ARCSEC])
                        v_lo = km.vrot_arctan(r_arcsec, None, None, [max(p_best - p_err, 1e-12), DISC_RT_ARCSEC])
                        v_err = 0.5 * np.abs(v_hi - v_lo)
                    else:
                        v_err = np.full_like(v, np.nan)

                    disc_best2_for_plots = {
                        "beta": np.full(num_shells_disc_eff, float(disc_fit["beta_best"])),
                        "beta_err": np.full(num_shells_disc_eff, best.get("beta_err", np.nan)),
                        "v": np.asarray(v, float),
                        "v_err": np.asarray(v_err, float),
                    }
                else:
                    disc_best2_for_plots = km._disc_profile_from_physical_mode(
                        fit_mode=DISC_FIT_MODE,
                        best_param=p_best,
                        best_param_err=p_err,
                        n_shells=num_shells_disc_eff,
                        rin_pix=rin_pix_disc,
                        rout_pix=rout_pix_disc,
                        arcsec_per_pix=arcsec_per_pix,
                        scale_kpc_per_arcsec=scale,
                        R_nsc_pc=DISC_R_NSC_PC,
                        a_plu_pc=DISC_A_PLU_PC,
                    )
                    disc_best2_for_plots["beta"] = np.full(num_shells_disc_eff, float(disc_fit["beta_best"]))
                    disc_best2_for_plots["beta_err"] = np.full(num_shells_disc_eff, best.get("beta_err", np.nan))

            else:
                # old independent-shell behavior
                if USE_GLOBAL_BETA_DISC:
                    best1 = disc_fit["best"]
                    beta_fixed = float(disc_fit["beta_best"])
                    beta_err_scalar = best1.get("beta_err_scalar", np.nan)

                    disc_best2_for_plots = km.summarize_fixed_beta_per_shell_v(
                        disc_fit["chi_squared_map"], disc_fit["beta_array"], disc_fit["v_array"],
                        beta_fixed, beta_err_scalar=beta_err_scalar
                    )
                else:
                    disc_best2_for_plots = km.summarize_free_beta_per_shell(
                        disc_fit["chi_squared_map"], disc_fit["beta_array"], disc_fit["v_array"],
                        delta_chi2=2.30
                    )

                _ = km.percentile_scatter_per_shell_best(
                    best=disc_best2_for_plots, obs=obs_disc_fit, vel_axis=vel,
                    center_xy=origin, pa_deg=gamma_disc,
                    n_shells=num_shells_disc_eff, r_min_pix=rin_pix_disc, r_max_pix=rout_pix_disc,
                    aperture_deg=disc_aperture, double_cone=disc_double_cone,
                    pixscale=pixscale, nrebin=nrebin, scale=scale,
                    min_pixels_per_shell=2, perc=perc_disc, ncloud=npt
                )

        except Exception as e:
            logger.warning("DISC summarize/scatter failed: %r", e)
            disc_best2_for_plots = None





        # build disc model cube (UNWEIGHTED copy for outflow context)
        if DISC_FIT_MODE in {"disk_kepler", "NSC", "Plummer", "disk_arctan"} or USE_GLOBAL_BETA_DISC:
            model_disc_best = km.build_best_model_from_fit(
                beta_best=float(disc_fit["beta_best"]),
                v_best=float(disc_fit["v_best"]),
                FIT_MODE=DISC_FIT_MODE,
                R_nsc=DISC_R_NSC_PC,
                a_plu=DISC_A_PLU_PC,
                rt=DISC_RT_ARCSEC,
            )
        else:
            if disc_best2_for_plots is None:
                raise RuntimeError("disc_best2_for_plots is None but USE_GLOBAL_BETA_DISC=False")

            radius_shells_disc = km._as_shell_ranges(radius_range_model_disc, num_shells_disc_eff)
            beta_arr_disc = list(np.asarray(disc_best2_for_plots["beta"], float))
            v_arr_disc    = list(np.asarray(disc_best2_for_plots["v"], float))

            model_disc_best = km._make_multishell_component(
                npt_total=int(npt),
                n_shells=num_shells_disc_eff,
                geometry="cylindrical",
                radius_range_shells=radius_shells_disc,
                theta_range=disc_theta_range,
                phi_range=disc_phi_range,
                zeta_range=disc_zeta_range,
                logradius=logradius,
                flux_func=None,
                vel1_func=km.vout, vel2_func=km.vout, vel3_func=km.vout,
                vel_sigma=vel_sigma,
                psf_sigma=psf_sigma,
                lsf_sigma=lsf_sigma,
                cube_range=cube_range,
                cube_nbins=cube_nbins,
                fluxpars=[1, 0.05/scale],
                v_arr=v_arr_disc,
                beta_arr=beta_arr_disc,
                xycenter=xy_AGN,
                alpha=0.0,
                gamma=float(gamma_disc),
                vsys=0.0
            )


        model_disc_best.generate_cube()
        disc_cube_for_outflow = np.array(model_disc_best.cube["data"], copy=True)

        # weighted disc cube for disc diagnostics / overlay
        model_disc_best.weight_cube(obs_disc_fit.cube["data"])
        model_disc_best.generate_cube(weights=model_disc_best.cube["weights"])

        # overlay
        try:
            inc_for_overlay = float(disc_fit["beta_best"]) if USE_GLOBAL_BETA_DISC else float(np.nanmedian(disc_best2_for_plots["beta"]))
            if DISC_FIT_MODE == "disk_kepler":
                disc_title = fr"DISC: β={disc_fit['beta_best']:.1f}°, $M_\bullet$={disc_fit['v_best']:.3e} $M_\odot$"
            elif DISC_FIT_MODE == "NSC":
                disc_title = fr"DISC: β={disc_fit['beta_best']:.1f}°, $A$={disc_fit['v_best']:.3e}"
            elif DISC_FIT_MODE == "Plummer":
                disc_title = fr"DISC: β={disc_fit['beta_best']:.1f}°, $M_0$={disc_fit['v_best']:.3e} $M_\odot$"
            elif DISC_FIT_MODE == "disk_arctan":
                disc_title = fr"DISC: β={disc_fit['beta_best']:.1f}°, $V_{{\max}}$={disc_fit['v_best']:.1f} km s$^{{-1}}$"
            else:
                disc_title = fr"DISC: β={disc_fit['beta_best']:.1f}°, v={disc_fit['v_best']:.0f} km s$^{{-1}}$"




            km.show_shells_overlay(
                cube_obs=obs_disc_fit.cube["data"], cube_model=model_disc_best.cube["data"],
                center_xy=origin, inc_deg=inc_for_overlay, pa_deg=float(gamma_disc),
                n_shells=num_shells_disc, r_min_pix=rin_pix_disc, r_max_pix=rout_pix_disc,
                aperture_deg=disc_aperture, double_cone=disc_double_cone,
                pixscale=pixscale, nrebin=nrebin, scale=scale,
                mask_mode="model", edges_mode="model",
                title= disc_title,
                debug_intrinsic=(disc_geometry.lower() == "cylindrical"),
                xlimit=xrange, ylimit=yrange
            )
            finalize_figure(output_dir / "04_disc_shell_overlay.png", show=cfg.output.show_plots)

        except Exception as e:
            logger.warning("DISC overlay failed: %r", e)


        disc_y_label = {
            "disk_kepler": r"$M_\bullet$ ($M_\odot$)",
            "NSC": r"$A$",
            "Plummer": r"$M_0$ ($M_\odot$)",
            "disk_arctan": r"$V_{\max}$ (km s$^{-1}$)",
        }.get(DISC_FIT_MODE, r"$v$ (km s$^{-1}$)")

        km.plot_residual_maps_cone(
            disc_fit["chi_squared_map"], disc_fit["beta_array"], disc_fit["v_array"],
            num_shells_disc_eff, best=disc_best2_for_plots, y_label= disc_y_label
        )
        finalize_figure(output_dir / "05_disc_chi2_maps.png", show=cfg.output.show_plots)

        try:
            fig, _, _ = km.inspect_percentiles_at(
                float(disc_fit["beta_best"]),
                float(disc_fit["v_best"]),
                perc=perc_disc,
                perc_weights=perc_weights,
                sigma_perc_kms=SIGMA_PERC_KMS,
                loss=loss,
                qgrid=CRPS_QGRID
            )
            if fig is not None:
                finalize_figure(output_dir / "06_disc_percentiles.png", show=cfg.output.show_plots)
            else:
                logger.warning("DISC inspect produced no figure; skipping 06_disc_percentiles.png")
        except Exception as e:
            logger.warning("DISC inspect failed: %r", e)



            
            
        # ============================================================
        # DISC-ONLY: moment maps comparison 
        # ============================================================
        if (FIT_COMPONENT_MODE == "disk") and (DO_FINAL_COMBINED_MODEL_PLOT or SAVE_ALL_OUTPUTS) and (model_disc_best is not None):
            m_disc_final = model_disc_best
            # Ensure we have model kinematic maps computed
            try:
                m_disc_final.generate_cube()
                m_disc_final.kin_maps(domap="all", fluxthr=np.nanpercentile(obscube, 1))
        
                m_disc_final.weight_cube(obs.cube["data"])
                m_disc_final.generate_cube(weights=m_disc_final.cube["weights"])
                m_disc_final.kin_maps_cube(fluxthr=np.nanpercentile(obscube, 1))
            except Exception as e:
                logger.warning("DISC-only maps build: %r", e)


            final_model = m_disc_final
            final_model_name = "bestfit_disk_weighted_cube.fits"


            # 3x3 comparison
            km.plot_kin_maps_3x3(
                obs=obs,
                m=m_disc_final,
                xy_AGN=xy_AGN,
                xrange=xrange,
                yrange=yrange,
                vrange=velrange,
                sigrange=sigrange,
                resid_ranges=resid_ranges,
                nticks=4,
                psf_bmaj=psf_sigma,
                psf_bmin=psf_sigma,
                psf_pa=12
            )
            finalize_figure(output_dir / "012a_mom_maps_comparison_best_fit.png", show=cfg.output.show_plots)



    # ==========================
    # 3) Combined context: include disc model in outflow fits
    # ==========================
    if FIT_COMPONENT_MODE == "disk_then_outflow":
        if disc_cube_for_outflow is None:
            raise RuntimeError("disc_cube_for_outflow is None: cannot fit outflow in disk_then_outflow mode.")
        km.set_fit_context(disc_cube=disc_cube_for_outflow)


    obs_out_pos = None
    obs_out_neg = None
    if FIT_COMPONENT_MODE in ("outflow", "disk_then_outflow"):
        cube_pos = _keep_only_mask(obs.cube["data"], mask_cone_pos, mode="nan")
        cube_neg = _keep_only_mask(obs.cube["data"], mask_cone_neg, mode="nan")
        obs_out_pos = km.make_observed_like(obs, cube_pos)
        obs_out_neg = km.make_observed_like(obs, cube_neg)

    # =========================================================================
    # 4) OUTFLOW FITS: lobe + and lobe - separately (FULL diagnostics for each)
    # =========================================================================
    outflow_fit_pos = None
    outflow_fit_neg = None
    model_outflow_pos_best = None
    model_outflow_neg_best = None
    out_best2_pos = None
    out_best2_neg = None

    def _run_single_lobe_outflow_fit(*, obs_lobe, pa_deg, label):
        """Run gridsearch + all diagnostics for one lobe; returns (fit_dict, model_best, best2_for_plots)."""
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
            .replace("-", "minus")
        )
        fit = km.fit_gridsearch_component(
            obs_for_fit=obs_lobe,
            disc_cube=None,
            vel_axis=vel,
            origin=origin,
            pixscale=pixscale,
            nrebin=nrebin,
            scale=scale,
            geometry=out_geometry,
            FIT_MODE=OUTFLOW_FIT_MODE,
            gamma_model_deg=float(pa_deg),
            aperture_deg=OUTFLOW_OPENING_DEG,
            double_cone=False,  # IMPORTANT: one lobe only
            radius_range_model_arcsec=radius_range_model_out,
            theta_range=[[0, round(OUTFLOW_OPENING_DEG/2)]],
            phi_range=[[0, 360]],
            zeta_range=[-0.1, 0.1],
            logradius=logradius,
            psf_sigma=psf_sigma,
            lsf_sigma=lsf_sigma,
            vel_sigma=vel_sigma,
            npt=npt,
            num_shells=num_shells_out,
            perc=perc_out,
            perc_weights=perc_weights,
            loss=loss,
            CRPS_QGRID=CRPS_QGRID,
            SIGMA_PERC_KMS=SIGMA_PERC_KMS,
            beta_min=beta_min_o, beta_max=beta_max_o, step_beta=step_beta_o,
            v_min=v_min_o, v_max=v_max_o, step_v=step_v_o,
            verbose_label=label
            )
        best_info = km._extract_best_fit_with_uncertainties(fit)
        logger.info(
            "%s Global best: beta=%.1f ± %.1f deg, v=%.0f ± %.0f km/s",
             label,
             best_info["beta_best"],
             best_info["beta_err"],
             best_info["v_best"],
             best_info["v_err"],
             )

        n_shells_eff = int(np.shape(fit["chi_squared_map"])[0])

        # global chi2(beta)
        beta_best_global, _, _, _ = km.plot_chi2_vs_beta_global(
            fit["chi_squared_map"],
            fit["beta_array"],
            fit["v_array"],
            USE_GLOBAL_BETA=USE_GLOBAL_BETA_OUT,
            reduce_v="min",
            combine_shells="sum",
            logy=True,
            title=fr"{label}: global $\chi^2$ vs $\beta$"
        )
        finalize_figure(output_dir / f"07_{safe_label}_global_beta_chi2.png", show=cfg.output.show_plots)
        if USE_GLOBAL_BETA_OUT:
            fit["beta_best"] = float(beta_best_global)

        # summarize
        try:
            if USE_GLOBAL_BETA_OUT:
                best1 = fit["best"]
                beta_fixed = float(best1.get("beta_star", fit["beta_best"]))
                beta_err_scalar = best1.get("beta_err_scalar", np.nan)

                best2 = km.summarize_fixed_beta_per_shell_v(
                    fit["chi_squared_map"], fit["beta_array"], fit["v_array"],
                    beta_fixed, beta_err_scalar=beta_err_scalar
                )
            else:
                best2 = km.summarize_free_beta_per_shell(
                    fit["chi_squared_map"], fit["beta_array"], fit["v_array"],
                    delta_chi2=2.30
                )
        except Exception as e:
            logger.warning("Label %r summarize %r", label, e)

            best2 = None

        # build best model cube
        if USE_GLOBAL_BETA_OUT:
            model_best = km.build_best_model_from_fit(
                beta_best=float(fit["beta_best"]),
                v_best=float(fit["v_best"]),
                FIT_MODE=OUTFLOW_FIT_MODE
            )
        else:
            if best2 is None:
                raise RuntimeError(f"{label}: best2 is None but USE_GLOBAL_BETA_OUT=False")

            radius_shells_out = km._as_shell_ranges(radius_range_model_out, n_shells_eff)
            beta_arr_tmp = list(np.asarray(best2["beta"], float))
            v_arr_tmp    = list(np.asarray(best2["v"], float))

            model_best = km._make_multishell_component(
                npt_total=int(npt),
                n_shells=n_shells_eff,
                geometry="spherical",
                radius_range_shells=radius_shells_out,
                theta_range=[[0, OUTFLOW_OPENING_DEG/2.0]],
                phi_range=[[0, 360]],
                zeta_range=[-0.1, 0.1],
                logradius=logradius,
                flux_func=None,
                vel1_func=km.vout, vel2_func=km.vout, vel3_func=km.vout,
                vel_sigma=vel_sigma,
                psf_sigma=psf_sigma,
                lsf_sigma=lsf_sigma,
                cube_range=cube_range,
                cube_nbins=cube_nbins,
                fluxpars=[1, 0.05/scale],
                v_arr=v_arr_tmp,
                beta_arr=beta_arr_tmp,
                xycenter=xy_AGN,
                alpha=0.0,
                gamma=float(pa_deg),
                vsys=0.0
            )

        model_best.generate_cube()

        # diagnostics: percentile scatter + overlay + chi2 maps + inspect
        try:
            if best2 is not None:
                _ = km.percentile_scatter_per_shell_best(
                    best=best2, obs=obs_lobe, vel_axis=vel,
                    center_xy=origin, pa_deg=float(pa_deg),
                    n_shells=n_shells_eff, r_min_pix=rin_pix_out, r_max_pix=rout_pix_out,
                    aperture_deg=OUTFLOW_OPENING_DEG, double_cone=False,
                    pixscale=pixscale, nrebin=nrebin, scale=scale,
                    min_pixels_per_shell=2, perc=perc_out, ncloud=npt
                )
        except Exception as e:
            logger.warning("label %r scatter %r", label, e)



        try:
            km.show_shells_overlay(
                cube_obs=obs_lobe.cube["data"], cube_model=model_best.cube["data"],
                center_xy=origin, inc_deg=float(fit["beta_best"]), pa_deg=float(pa_deg),
                n_shells=num_shells_out, r_min_pix=rin_pix_out, r_max_pix=rout_pix_out,
                aperture_deg=OUTFLOW_OPENING_DEG, double_cone=False,
                pixscale=pixscale, nrebin=nrebin, scale=scale,
                mask_mode="model", edges_mode="model",
                title=fr"{label}: β={fit['beta_best']:.1f}°, v={fit['v_best']:.0f} km s$^{{-1}}$",
                debug_intrinsic=True,
                xlimit=xrange, ylimit=yrange
            )
            finalize_figure(output_dir / f"08_{safe_label}_shell_overlay.png", show=cfg.output.show_plots)

        except Exception as e:
            logger.warning("label %r overlay %r", label, e)

        km.plot_residual_maps_cone(
            fit["chi_squared_map"], fit["beta_array"], fit["v_array"],
            n_shells_eff, best=best2, y_label=r"$v$ (km s$^{-1}$)"
        )
        finalize_figure(output_dir / f"09_{safe_label}_chi2_maps.png", show=cfg.output.show_plots)

        try:
            fig, _, _ = km.inspect_percentiles_at(
                float(fit["beta_best"]),
                float(fit["v_best"]),
                perc=perc_out,
                perc_weights=perc_weights,
                sigma_perc_kms=SIGMA_PERC_KMS,
                loss=loss,
                qgrid=CRPS_QGRID
            )
            if fig is not None:
                finalize_figure(output_dir / f"10_{safe_label}_percentiles.png", show=cfg.output.show_plots)
            else:
                logger.warning("label %r inspect produced no figure; skipping percentile plot", label)

        except Exception as e:
            logger.warning("label %r inspect %r", label, e)



        return fit, model_best, best2


    if FIT_COMPONENT_MODE in ("outflow", "disk_then_outflow"):

        if FIT_COMPONENT_MODE == "outflow":
            km.set_fit_context(disc_cube=None)

        fit_bicone = (str(OUTFLOW_MASK_MODE).lower() == "bicone") or bool(OUTFLOW_DOUBLE_CONE)
        gamma_neg = (OUTFLOW_PA_DEG + 180.0) % 360.0

        if fit_bicone:
            logger.info(
                "OUTFLOW bicone mode: fitting both lobes (PA=%.1f deg and %.1f deg, aperture=%.1f deg)",
                OUTFLOW_PA_DEG,
                gamma_neg,
                OUTFLOW_OPENING_DEG,
            )

            logger.info(
                "OUTFLOW lobe +: PA=%.1f deg, aperture=%.1f deg",
                OUTFLOW_PA_DEG,
                OUTFLOW_OPENING_DEG,
            )
            outflow_fit_pos, model_outflow_pos_best, out_best2_pos = _run_single_lobe_outflow_fit(
                obs_lobe=obs_out_pos,
                pa_deg=OUTFLOW_PA_DEG,
                label="OUTFLOW (+)"
            )

            logger.info(
                "OUTFLOW lobe -: PA=%.1f deg, aperture =%.1f deg",
                gamma_neg,
                OUTFLOW_OPENING_DEG,
            )
            outflow_fit_neg, model_outflow_neg_best, out_best2_neg = _run_single_lobe_outflow_fit(
                obs_lobe=obs_out_neg,
                pa_deg=gamma_neg,
                label="OUTFLOW (-)"
            )

        else:
            if OUTFLOW_AXIS_SIGN >= 0:
                logger.info(
                    "OUTFLOW single-cone mode: fitting only lobe + (PA=%.1f deg, aperture=%.1f deg)",
                    OUTFLOW_PA_DEG,
                    OUTFLOW_OPENING_DEG,
                )
                outflow_fit_pos, model_outflow_pos_best, out_best2_pos = _run_single_lobe_outflow_fit(
                    obs_lobe=obs_out_pos,
                    pa_deg=OUTFLOW_PA_DEG,
                    label="OUTFLOW (+)"
                )
                outflow_fit_neg = None
                model_outflow_neg_best = None
                out_best2_neg = None

            else:
                logger.info(
                    "OUTFLOW single-cone mode: fitting only lobe - (PA=%.1f deg, aperture=%.1f deg)",
                    gamma_neg,
                    OUTFLOW_OPENING_DEG,
                )
                outflow_fit_neg, model_outflow_neg_best, out_best2_neg = _run_single_lobe_outflow_fit(
                    obs_lobe=obs_out_neg,
                    pa_deg=gamma_neg,
                    label="OUTFLOW (-)"
                )
                outflow_fit_pos = None
                model_outflow_pos_best = None
                out_best2_pos = None

    # ============================================================
    # OUTFLOW-ONLY: build total outflow model 
    # ============================================================
    if (FIT_COMPONENT_MODE == "outflow") and (DO_FINAL_COMBINED_MODEL_PLOT or SAVE_ALL_OUTPUTS):

        
        build_bicone_total = (str(OUTFLOW_MASK_MODE).lower() == "bicone") or bool(OUTFLOW_DOUBLE_CONE)

        xycenter = xy_AGN
        alpha = 0.0
        vsys = 0.0
        expradius = 0.05
        fluxpars = [1, expradius / scale]
        
        flux_func = None
        vel1_func = km.vout
        vel2_func = km.vout
        vel3_func = km.vout
        

        radius_shells_out = km._as_shell_ranges(radius_range_model_out, num_shells_out)
        theta_out = [[0, OUTFLOW_OPENING_DEG/2.0]]
        phi_out   = [[0, 360]]
        zeta_out  = [-0.1, 0.1]

        def _build_outflow_lobe_from_fit(fit_dict, gamma_deg):
            """Build a lobe model from a fit result (global-beta or free-beta)."""
            if fit_dict is None:
                return None

            beta_best = float(fit_dict["beta_best"])
            v_best    = float(fit_dict["v_best"])

            if num_shells_out <= 1:
                mm = km._make_single_km_component(
                    npt=int(npt)*100,
                    geometry="spherical",
                    radius_range=radius_range_model_out,
                    theta_range=theta_out,
                    phi_range=phi_out,
                    zeta_range=zeta_out,
                    logradius=logradius,
                    flux_func=flux_func,
                    vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
                    vel_sigma=vel_sigma,
                    psf_sigma=psf_sigma,
                    lsf_sigma=lsf_sigma,
                    cube_range=cube_range,
                    cube_nbins=cube_nbins,
                    fluxpars=fluxpars,
                    v=v_best,
                    xycenter=xycenter,
                    alpha=alpha,
                    beta=beta_best,
                    gamma=float(gamma_deg),
                    vsys=vsys
                )
            else:
                # If USE_GLOBAL_BETA_OUT 
                if USE_GLOBAL_BETA_OUT:
                    beta_arr = [beta_best] * num_shells_out
                    v_arr    = [v_best]    * num_shells_out
                else:
                    if abs((gamma_deg - OUTFLOW_PA_DEG) % 360.0) < 1e-6:
                        b2 = out_best2_pos
                    else:
                        b2 = out_best2_neg
                    if b2 is None:
                        beta_arr = [beta_best] * num_shells_out
                        v_arr    = [v_best]    * num_shells_out
                    else:
                        beta_arr = list(np.asarray(b2["beta"], float))
                        v_arr    = list(np.asarray(b2["v"], float))

                mm = km._make_multishell_component(
                    npt_total=int(npt)*100,
                    n_shells=num_shells_out,
                    geometry="spherical",
                    radius_range_shells=radius_shells_out,
                    theta_range=theta_out,
                    phi_range=phi_out,
                    zeta_range=zeta_out,
                    logradius=logradius,
                    flux_func=flux_func,
                    vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
                    vel_sigma=vel_sigma,
                    psf_sigma=psf_sigma,
                    lsf_sigma=lsf_sigma,
                    cube_range=cube_range,
                    cube_nbins=cube_nbins,
                    fluxpars=fluxpars,
                    v_arr=v_arr,
                    beta_arr=beta_arr,
                    xycenter=xycenter,
                    alpha=alpha,
                    gamma=float(gamma_deg),
                    vsys=vsys
                )

            if not hasattr(mm, "zeta"):
                mm.zeta = np.zeros_like(mm.x, dtype=float)
            return mm

        # Create a "total outflow model" container by starting from one lobe and add the second if requested
        gamma_pos = float(OUTFLOW_PA_DEG)
        gamma_neg = float((OUTFLOW_PA_DEG + 180.0) % 360.0)

        if build_bicone_total:
            if (outflow_fit_pos is None) or (outflow_fit_neg is None):
                logger.warning("bicone requested but one lobe fit is missing; skipping outflow-only 3x3")



                m_out_total = None
            else:
                m_out_total = _build_outflow_lobe_from_fit(outflow_fit_pos, gamma_pos)
                m2 = _build_outflow_lobe_from_fit(outflow_fit_neg, gamma_neg)
                if (m_out_total is not None) and (m2 is not None):
                    m_out_total.add_model(m2)
        else:
            # Single cone: use sign to choose which one is "the outflow"
            if OUTFLOW_AXIS_SIGN >= 0:
                m_out_total = _build_outflow_lobe_from_fit(outflow_fit_pos, gamma_pos)
            else:
                m_out_total = _build_outflow_lobe_from_fit(outflow_fit_neg, gamma_neg)

        # Generate maps and plot 3x3
        if m_out_total is not None:
            try:
                m_out_total.generate_cube()
                m_out_total.kin_maps(domap="all", fluxthr=np.nanpercentile(obscube, 1))

                m_out_total.weight_cube(obs.cube["data"])
                m_out_total.generate_cube(weights=m_out_total.cube["weights"])
                m_out_total.kin_maps_cube(fluxthr=np.nanpercentile(obscube, 1))
            except Exception as e:
                logger.warning("OUTFLOW-only maps build %r", e)


            final_model = m_out_total
            if build_bicone_total:
                final_model_name = "bestfit_outflow_bicone_weighted_cube.fits"
            else:
                final_model_name = "bestfit_outflow_singlecone_weighted_cube.fits"

            km.plot_kin_maps_3x3(
                obs=obs,
                m=m_out_total,
                xy_AGN=xy_AGN,
                xrange=xrange,
                yrange=yrange,
                vrange=velrange,
                sigrange=sigrange,
                resid_ranges=resid_ranges,
                nticks=4,
                psf_bmaj=psf_sigma,
                psf_bmin=psf_sigma,
                psf_pa=12
            )
            finalize_figure(output_dir / "012b_mom_maps_comparison_best_fit.png", show=cfg.output.show_plots)

    # =============================================================
    # 5) Final “pack masks” inspection (DISC, OUTFLOW +, OUTFLOW -)
    # =============================================================
    try:
        if FIT_COMPONENT_MODE in ("disk", "disk_then_outflow") and (model_disc_best is not None):
            _ = km.residuals_percentiles_cone(
                cube_model=model_disc_best.cube["data"],
                cube_obs=obs_disc_fit.cube["data"] if obs_disc_fit is not None else obs.cube["data"],
                vel_axis=vel, center_xy=origin,
                inc_deg=float(disc_fit["beta_best"]), pa_deg=float(gamma_disc),
                n_shells=num_shells_disc, r_min_pix=rin_pix_disc, r_max_pix=rout_pix_disc,
                aperture_deg=disc_aperture, double_cone=disc_double_cone,
                perc=perc_disc, sigma_perc_kms=SIGMA_PERC_KMS,
                mask_mode="model", edges_mode="model",
                min_pixels_per_shell=2,
                perc_weights=1.0,
                loss=loss, qgrid=CRPS_QGRID
            )

        if (model_outflow_pos_best is not None) and (outflow_fit_pos is not None):
            _ = km.residuals_percentiles_cone(
                cube_model=model_outflow_pos_best.cube["data"],
                cube_obs=obs_out_pos.cube["data"],
                vel_axis=vel, center_xy=origin,
                inc_deg=float(outflow_fit_pos["beta_best"]), pa_deg=float(OUTFLOW_PA_DEG),
                n_shells=num_shells_out, r_min_pix=rin_pix_out, r_max_pix=rout_pix_out,
                aperture_deg=OUTFLOW_OPENING_DEG, double_cone=False,
                perc=perc_out, sigma_perc_kms=SIGMA_PERC_KMS,
                mask_mode="model", edges_mode="model",
                min_pixels_per_shell=2,
                perc_weights=1.0,
                loss=loss, qgrid=CRPS_QGRID
            )

        if (model_outflow_neg_best is not None) and (outflow_fit_neg is not None):
            gamma_neg = (OUTFLOW_PA_DEG + 180.0) % 360.0
            _ = km.residuals_percentiles_cone(
                cube_model=model_outflow_neg_best.cube["data"],
                cube_obs=obs_out_neg.cube["data"],
                vel_axis=vel, center_xy=origin,
                inc_deg=float(outflow_fit_neg["beta_best"]), pa_deg=float(gamma_neg),
                n_shells=num_shells_out, r_min_pix=rin_pix_out, r_max_pix=rout_pix_out,
                aperture_deg=OUTFLOW_OPENING_DEG, double_cone=False,
                perc=perc_out, sigma_perc_kms=SIGMA_PERC_KMS,
                mask_mode="model", edges_mode="model",
                min_pixels_per_shell=2,
                perc_weights=1.0,
                loss=loss, qgrid=CRPS_QGRID
            )
    except Exception as e:
        logger.warning("Pack masks %r", e)




    # ============================================================
    # 6) FINAL COMBINED MODEL (DISC + OUTFLOW+ + OUTFLOW-) + 3x3 moment comparison
    # ============================================================
    if (DO_FINAL_COMBINED_MODEL_PLOT or SAVE_ALL_OUTPUTS) and \
       (model_disc_best is not None) and \
       (model_outflow_pos_best is not None) and \
       (model_outflow_neg_best is not None):

        logger.info("\n--- Building final combined (DISC + OUTFLOW+ + OUTFLOW-) model ---")

        # best-fit parameters
        beta_disc_best  = float(disc_fit["beta_best"])
        v_disc_best     = float(disc_fit["v_best"])
        gamma_disc_best = float(gamma_disc)

        beta_out_pos_best = float(outflow_fit_pos["beta_best"])
        v_out_pos_best    = float(outflow_fit_pos["v_best"])
        gamma_out_pos_best = float(OUTFLOW_PA_DEG)

        gamma_neg = (OUTFLOW_PA_DEG + 180.0) % 360.0
        beta_out_neg_best = float(outflow_fit_neg["beta_best"])
        v_out_neg_best    = float(outflow_fit_neg["v_best"])
        gamma_out_neg_best = float(gamma_neg)
        # common
        xycenter = xy_AGN
        alpha = 0.0
        vsys = 0.0
        flux_func = None
        vel1_func = km.vout
        vel2_func = km.vout
        vel3_func = km.vout
        expradius = 0.05
        fluxpars = [1, expradius / scale]

        # ---- DISC component ----

        if DISC_FIT_MODE in {"disk_kepler", "NSC", "Plummer", "disk_arctan"}:
            if model_disc_best is None:
                raise RuntimeError("model_disc_best is None in final combined model build.")
            m_final = copy.deepcopy(model_disc_best)
        else:
            radius_shells_disc = km._as_shell_ranges(radius_range_model_disc, num_shells_disc_eff)

            if USE_GLOBAL_BETA_DISC:
                beta_arr_disc = [beta_disc_best] * num_shells_disc_eff
                v_arr_disc    = [v_disc_best]    * num_shells_disc_eff
            else:
                beta_arr_disc = list(np.asarray(disc_best2_for_plots["beta"], float))
                v_arr_disc    = list(np.asarray(disc_best2_for_plots["v"], float))

            m_final = km._make_multishell_component(
                npt_total=int(npt)*100,
                n_shells=num_shells_disc_eff,
                geometry="cylindrical",
                radius_range_shells=radius_range_model_disc if num_shells_disc_eff == 1 else radius_shells_disc,
                theta_range=disc_theta_range,
                phi_range=disc_phi_range,
                zeta_range=disc_zeta_range,
                logradius=logradius,
                flux_func=flux_func,
                vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
                vel_sigma=vel_sigma,
                psf_sigma=psf_sigma,
                lsf_sigma=lsf_sigma,
                cube_range=cube_range,
                cube_nbins=cube_nbins,
                fluxpars=fluxpars,
                v_arr=v_arr_disc,
                beta_arr=beta_arr_disc,
                xycenter=xycenter,
                alpha=alpha,
                gamma=gamma_disc_best,
                vsys=vsys
            )



            if USE_GLOBAL_BETA_DISC:
                beta_arr_disc = [beta_disc_best] * num_shells_disc_eff
                v_arr_disc    = [v_disc_best]    * num_shells_disc_eff
            else:
                beta_arr_disc = list(np.asarray(disc_best2_for_plots["beta"], float))
                v_arr_disc    = list(np.asarray(disc_best2_for_plots["v"], float))

            m_final = km._make_multishell_component(
                npt_total=int(npt)*100,
                n_shells=num_shells_disc_eff,
                geometry="cylindrical",
                radius_range_shells=radius_range_model_disc if num_shells_disc_eff == 1 else radius_shells_disc,
                theta_range=disc_theta_range,
                phi_range=disc_phi_range,
                zeta_range=disc_zeta_range,
                logradius=logradius,
                flux_func=flux_func,
                vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
                vel_sigma=vel_sigma,
                psf_sigma=psf_sigma,
                lsf_sigma=lsf_sigma,
                cube_range=cube_range,
                cube_nbins=cube_nbins,
                fluxpars=fluxpars,
                v_arr=v_arr_disc,
                beta_arr=beta_arr_disc,
                xycenter=xycenter,
                alpha=alpha,
                gamma=gamma_disc_best,
                vsys=vsys
            )


        # ---- OUTFLOW components  ----
        radius_shells_out = km._as_shell_ranges(radius_range_model_out, num_shells_out)
        theta_out = [[0, OUTFLOW_OPENING_DEG/2.0]]
        phi_out   = [[0, 360]]
        zeta_out  = [-0.1, 0.1]

        def _build_outflow_lobe(beta_best, v_best, gamma_best):
            if num_shells_out <= 1:
                mm = km._make_single_km_component(
                    npt=int(npt)*100,
                    geometry="spherical",
                    radius_range=radius_range_model_out,
                    theta_range=theta_out,
                    phi_range=phi_out,
                    zeta_range=zeta_out,
                    logradius=logradius,
                    flux_func=flux_func,
                    vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
                    vel_sigma=vel_sigma,
                    psf_sigma=psf_sigma,
                    lsf_sigma=lsf_sigma,
                    cube_range=cube_range,
                    cube_nbins=cube_nbins,
                    fluxpars=fluxpars,
                    v=float(v_best),
                    xycenter=xycenter,
                    alpha=alpha,
                    beta=float(beta_best),
                    gamma=float(gamma_best),
                    vsys=vsys
                )
            else:
                mm = km._make_multishell_component(
                    npt_total=int(npt)*100,
                    n_shells=num_shells_out,
                    geometry="spherical",
                    radius_range_shells=radius_shells_out,
                    theta_range=theta_out,
                    phi_range=phi_out,
                    zeta_range=zeta_out,
                    logradius=logradius,
                    flux_func=flux_func,
                    vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
                    vel_sigma=vel_sigma,
                    psf_sigma=psf_sigma,
                    lsf_sigma=lsf_sigma,
                    cube_range=cube_range,
                    cube_nbins=cube_nbins,
                    fluxpars=fluxpars,
                    v_arr=[float(v_best)] * num_shells_out,
                    beta_arr=[float(beta_best)] * num_shells_out,
                    xycenter=xycenter,
                    alpha=alpha,
                    gamma=float(gamma_best),
                    vsys=vsys
                )

            if not hasattr(mm, "zeta"):
                mm.zeta = np.zeros_like(mm.x, dtype=float)
            return mm

        m_out_pos = _build_outflow_lobe(beta_out_pos_best, v_out_pos_best, gamma_out_pos_best)
        m_out_neg = _build_outflow_lobe(beta_out_neg_best, v_out_neg_best, gamma_out_neg_best)

        m_final.add_model(m_out_pos)
        m_final.add_model(m_out_neg)

        # generate + maps + final plot
        m_final.generate_cube()
        m_final.kin_maps(domap="all", fluxthr=np.nanpercentile(obscube, 1))

        m_final.weight_cube(obs.cube["data"])
        m_final.generate_cube(weights=m_final.cube["weights"])
        m_final.kin_maps_cube(fluxthr=np.nanpercentile(obscube, 1))


        final_model = m_final
        final_model_name = "bestfit_disk_outflow_weighted_cube.fits"

        km.plot_kin_maps_3x3(
            obs=obs,
            m=m_final,
            xy_AGN=xy_AGN,
            xrange=xrange,
            yrange=yrange,
            vrange=velrange,
            sigrange=sigrange,
            resid_ranges=resid_ranges,
            nticks=4,
            psf_bmaj=psf_sigma,
            psf_bmin=psf_sigma,
            psf_pa=12
        )
        finalize_figure(output_dir / "012c_mom_maps_comparison_best_fit.png", show=cfg.output.show_plots)



    # ==========================
    # Profiles: disc, outflow +, outflow - and combined comparison
    # ==========================
    best_disc_profile = None
    if disc_fit is not None:
        if DISC_FIT_MODE in {"disk_kepler", "NSC", "Plummer", "disk_arctan"}:
            best_disc_profile = disc_best2_for_plots
        else:
            best_disc_profile = disc_fit.get("best", None)

        km._plot_v_profile(
            best_disc_profile,
            n_shells=num_shells_disc_eff,
            title="Disc circular velocity profile",
            scale_kpc_per_arcsec=scale,
            rin_pix=rin_pix_disc, rout_pix=rout_pix_disc,
            arcsec_per_pix=arcsec_per_pix
        )
        finalize_figure(output_dir / "013_disc_vel_profiles.png", show=cfg.output.show_plots)


    best_out_pos_profile = None
    if outflow_fit_pos is not None:
        best_out_pos_profile = outflow_fit_pos.get("best", None)
        km._plot_v_profile(
            best_out_pos_profile,
            n_shells=num_shells_out,
            title="Outflow (+) velocity profile",
            scale_kpc_per_arcsec=scale,
            rin_pix=rin_pix_out, rout_pix=rout_pix_out,
            arcsec_per_pix=arcsec_per_pix
        )
        finalize_figure(output_dir / "013_out_plus_vel_profiles.png", show=cfg.output.show_plots)


    best_out_neg_profile = None
    if outflow_fit_neg is not None:
        best_out_neg_profile = outflow_fit_neg.get("best", None)
        km._plot_v_profile(
            best_out_neg_profile,
            n_shells=num_shells_out,
            title="Outflow (-) velocity profile",
            scale_kpc_per_arcsec=scale,
            rin_pix=rin_pix_out, rout_pix=rout_pix_out,
            arcsec_per_pix=arcsec_per_pix
        )
        finalize_figure(output_dir / "013_out_minus_vel_profiles.png", show=cfg.output.show_plots)


    if disc_best2_for_plots is not None and (not USE_GLOBAL_BETA_DISC):
        km.plot_beta_profile(disc_best2_for_plots, num_shells_disc_eff, "Disc inclination",
                             rin_pix_disc, rout_pix_disc, arcsec_per_pix, scale)
        finalize_figure(output_dir / "014_disc_beta_profile.png", show=cfg.output.show_plots)


    if out_best2_pos is not None and (not USE_GLOBAL_BETA_OUT):
        km.plot_beta_profile(out_best2_pos, int(np.shape(outflow_fit_pos["chi_squared_map"])[0]),
                             "Outflow (+) inclination", rin_pix_out, rout_pix_out, arcsec_per_pix, scale)
        finalize_figure(output_dir / "015_outflow_plus_beta_profile.png", show=cfg.output.show_plots)


    if out_best2_neg is not None and (not USE_GLOBAL_BETA_OUT):
        km.plot_beta_profile(out_best2_neg, int(np.shape(outflow_fit_neg["chi_squared_map"])[0]),
                             "Outflow (-) inclination", rin_pix_out, rout_pix_out, arcsec_per_pix, scale)
        finalize_figure(output_dir / "016_outflow_minus_beta_profile.png", show=cfg.output.show_plots)

    # ---- Combined comparison plot: Disc vs Outflow(+) vs Outflow(-) ----
    if (best_disc_profile is not None) and (best_out_pos_profile is not None) and (best_out_neg_profile is not None):

        vD = np.asarray(best_disc_profile.get("v", []), float)
        eD = np.asarray(best_disc_profile.get("v_err", np.full_like(vD, np.nan)), float)

        vP = np.asarray(best_out_pos_profile.get("v", []), float)
        eP = np.asarray(best_out_pos_profile.get("v_err", np.full_like(vP, np.nan)), float)

        vN = np.asarray(best_out_neg_profile.get("v", []), float)
        eN = np.asarray(best_out_neg_profile.get("v_err", np.full_like(vN, np.nan)), float)

        # shell edges and midpoints
        edges_pix_D = np.linspace(float(rin_pix_disc), float(rout_pix_disc), int(num_shells_disc) + 1)
        edges_arc_D = edges_pix_D * arcsec_per_pix
        rmid_arc_D  = 0.5 * (edges_arc_D[:-1] + edges_arc_D[1:])
        xerr_arc_D  = 0.5 * (edges_arc_D[1:] - edges_arc_D[:-1])

        edges_pix_O = np.linspace(float(rin_pix_out), float(rout_pix_out), int(num_shells_out) + 1)
        edges_arc_O = edges_pix_O * arcsec_per_pix
        rmid_arc_O  = 0.5 * (edges_arc_O[:-1] + edges_arc_O[1:])
        xerr_arc_O  = 0.5 * (edges_arc_O[1:] - edges_arc_O[:-1])

        rmax_arc = float(max(radius_range_model_disc[1], radius_range_model_out[1]))
        xmin, xmax = 0.0, rmax_arc
        pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.1
        xmin, xmax = max(0.0, xmin), xmax + pad

        fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="minor", length=3)
        ax.tick_params(axis="both", labelsize=12)

        ax.errorbar(rmid_arc_D, vD, xerr=xerr_arc_D, yerr=eD,
                    fmt="D-", mfc="white", mec="black", mew=1.5, lw=1.5, capsize=4, label="Disc")
        ax.errorbar(rmid_arc_O, vP, xerr=xerr_arc_O, yerr=eP,
                    fmt="o-", mfc="white", mec="black", mew=1.5, lw=1.5, capsize=4, label="Outflow (+)")
        ax.errorbar(rmid_arc_O, vN, xerr=xerr_arc_O, yerr=eN,
                    fmt="s-", mfc="white", mec="black", mew=1.5, lw=1.5, capsize=4, label="Outflow (-)")

        ax.set_xlabel(r"Radius [arcsec]", fontsize=14)
        ax.set_ylabel(r"Velocity [km s$^{-1}$]", fontsize=14)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(alpha=0.2)

        ax.set_xlim(xmin, xmax)

        def a2k(x):
            return x * float(scale)

        def k2a(x):
            return x / float(scale)

        secax = ax.secondary_xaxis("top", functions=(a2k, k2a))
        secax.set_xlabel("Radius [kpc]", fontsize=14)
        secax.tick_params(axis="both", labelsize=12)


        plt.tight_layout()
        finalize_figure(output_dir / "99_vel_profiles.png", show=cfg.output.show_plots)


    # ---- Combined comparison plot: Disc vs Outflow(+) vs Outflow(-) ----
    if (best_disc_profile is None) and (best_out_pos_profile is not None) and (best_out_neg_profile is not None):

        vP = np.asarray(best_out_pos_profile.get("v", []), float)
        eP = np.asarray(best_out_pos_profile.get("v_err", np.full_like(vP, np.nan)), float)

        vN = np.asarray(best_out_neg_profile.get("v", []), float)
        eN = np.asarray(best_out_neg_profile.get("v_err", np.full_like(vN, np.nan)), float)

        edges_pix_O = np.linspace(float(rin_pix_out), float(rout_pix_out), int(num_shells_out) + 1)
        edges_arc_O = edges_pix_O * arcsec_per_pix
        rmid_arc_O  = 0.5 * (edges_arc_O[:-1] + edges_arc_O[1:])
        xerr_arc_O  = 0.5 * (edges_arc_O[1:] - edges_arc_O[:-1])

        rmax_arc = float( radius_range_model_out[1])
        xmin, xmax = 0.0, rmax_arc
        pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.1
        xmin, xmax = max(0.0, xmin), xmax + pad

        fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="minor", length=3)
        ax.tick_params(axis="both", labelsize=12)

        ax.errorbar(rmid_arc_O, vP, xerr=xerr_arc_O, yerr=eP,
                    fmt="o-", mfc="white", mec="black", mew=1.5, lw=1.5, capsize=4, label="Outflow (+)")
        ax.errorbar(rmid_arc_O, vN, xerr=xerr_arc_O, yerr=eN,
                    fmt="s-", mfc="white", mec="black", mew=1.5, lw=1.5, capsize=4, label="Outflow (-)")

        ax.set_xlabel(r"Radius [arcsec]", fontsize=14)
        ax.set_ylabel(r"Velocity [km s$^{-1}$]", fontsize=14)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(alpha=0.2)

        ax.set_xlim(xmin, xmax)

        def a2k(x):
            return x * float(scale)

        def k2a(x):
            return x / float(scale)
        secax = ax.secondary_xaxis("top", functions=(a2k, k2a))
        secax.set_xlabel("Radius [kpc]", fontsize=14)
        secax.tick_params(axis="both", labelsize=12)


        plt.tight_layout()
        finalize_figure(output_dir / "99_vel_profiles.png", show=cfg.output.show_plots)


    # ============================================================
    # Outflow energetics: shell-by-shell M, Mdot, Pdot, Edot
    # Available only when outflow is part of the fit
    # ============================================================
    energetics_pos = None
    energetics_neg = None
    energetics_plot_path = None


    if COMPUTE_ENERGETICS:

        if FIT_COMPONENT_MODE == "disk":
            logger.warning(
                "compute_energetics=True but FIT_COMPONENT_MODE='disk'. "
                "Skipping outflow energetics."
            )

        else:
            line_id = km.identify_emission_line(
                cfg.line.wavelength_line,
                cfg.line.wavelength_line_unit
            )

            if line_id is None:
                logger.warning(
                    "compute_energetics=True but the selected emission line "
                    "(wavelength_line=%.2f %s) is not currently supported for energetics. "
                    "At the moment energetics are implemented only for [OIII]5007, Halpha, and Hbeta. "
                    "Skipping energetics calculation. This is still work in progress.",
                    float(cfg.line.wavelength_line),
                    str(cfg.line.wavelength_line_unit),
                )
            else:
                logger.info(
                    "Energetics enabled: recognized emission line = %s",
                    km.emission_line_label(line_id)
                                )
                logger.info(km.energetics_line_description(line_id))
                logger.info(
                    "Shell line fluxes will be computed as sum(cube) * d_lambda * unit_scale, "
                    "with unit_scale=%.6e from FITS BUNIT.",
                    flux_unit_scale,
                )


                lambda_obs_ang = (line_obs.to(u.AA)).value
                dv_kms = float(np.nanmedian(np.abs(np.diff(vel))))

                def _fallback_density_array(profile_len, ne_scalar):
                    return np.full(int(profile_len), float(ne_scalar), dtype=float)

                def _build_one_lobe_energetics(obs_lobe, best2_profile, sign_label, fallback_ne=None):
                    if (obs_lobe is None) or (best2_profile is None):
                        return None

                    vel_prof = km._extract_radial_profile(
                        best_profile=best2_profile,
                        rin_pix=rin_pix_out,
                        rout_pix=rout_pix_out,
                        n_shells=num_shells_out,
                        arcsec_per_pix=arcsec_per_pix,
                    )
                    if vel_prof is None:
                        return None

                    r_edges_pix = km.radial_shell_edges_pix(rin_pix_out, rout_pix_out, num_shells_out)
                    shell_masks = km.radial_shell_masks_yx(
                        shape_yx=obs_lobe.cube["data"].shape[1:],
                        center_xy=origin,
                        r_edges_pix=r_edges_pix,
                        extra_mask=None,
                    )

                    if ne_map_2d is not None:
                        ne_shell = km.shell_density_from_map(ne_map_2d, shell_masks, reducer="median")

                        if fallback_ne is not None:
                            bad = ~np.isfinite(ne_shell) | (ne_shell <= 0)
                            ne_shell[bad] = float(fallback_ne)

                        prof = km.build_outflow_energetics_profile(
                            cube_data=obs_lobe.cube["data"],
                            center_xy=origin,
                            rmin_pix=rin_pix_out,
                            rmax_pix=rout_pix_out,
                            n_shells=num_shells_out,
                            arcsec_per_pix=arcsec_per_pix,
                            scale_kpc_per_arcsec=scale,
                            dv_kms=dv_kms,
                            lambda_obs_angstrom=lambda_obs_ang,
                            luminosity_distance_mpc=D_L.to_value("Mpc"),
                            velocity_profile=vel_prof,
                            ne_shell=ne_shell,
                            line_id=line_id,
                            z_over_zsun=OIII_METALLICITY_Z_OVER_ZSUN,
                            flux_unit_scale=flux_unit_scale,
                        )
                        prof["density_mode"] = "map"
                        return prof

                    if fallback_ne is not None:
                        ne_shell = _fallback_density_array(len(vel_prof["v"]), fallback_ne)

                        prof = km.build_outflow_energetics_profile(
                            cube_data=obs_lobe.cube["data"],
                            center_xy=origin,
                            rmin_pix=rin_pix_out,
                            rmax_pix=rout_pix_out,
                            n_shells=num_shells_out,
                            arcsec_per_pix=arcsec_per_pix,
                            scale_kpc_per_arcsec=scale,
                            dv_kms=dv_kms,
                            lambda_obs_angstrom=lambda_obs_ang,
                            luminosity_distance_mpc=D_L.to_value("Mpc"),
                            velocity_profile=vel_prof,
                            ne_shell=ne_shell,
                            line_id=line_id,
                            z_over_zsun=OIII_METALLICITY_Z_OVER_ZSUN,
                            flux_unit_scale=flux_unit_scale,
                        )
                        prof["density_mode"] = "constant"
                        return prof

                    assumed_profiles = []
                    for ne_assumed in ASSUMED_NE_VALUES:
                        ne_shell = _fallback_density_array(len(vel_prof["v"]), ne_assumed)
                        pp = km.build_outflow_energetics_profile(
                            cube_data=obs_lobe.cube["data"],
                            center_xy=origin,
                            rmin_pix=rin_pix_out,
                            rmax_pix=rout_pix_out,
                            n_shells=num_shells_out,
                            arcsec_per_pix=arcsec_per_pix,
                            scale_kpc_per_arcsec=scale,
                            dv_kms=dv_kms,
                            lambda_obs_angstrom=lambda_obs_ang,
                            luminosity_distance_mpc=D_L.to_value("Mpc"),
                            velocity_profile=vel_prof,
                            ne_shell=ne_shell,
                            line_id=line_id,
                            z_over_zsun=OIII_METALLICITY_Z_OVER_ZSUN,
                            flux_unit_scale=flux_unit_scale,
                        )
                        assumed_profiles.append((float(ne_assumed), pp))

                    prof0 = copy.deepcopy(assumed_profiles[0][1])
                    mdot_stack = np.vstack([pp["mdot_msun_yr"] for _, pp in assumed_profiles])
                    mass_stack = np.vstack([pp["mass_msun"] for _, pp in assumed_profiles])
                    pdot_stack = np.vstack([pp["pdot_dyne"] for _, pp in assumed_profiles])
                    edot_stack = np.vstack([pp["edot_erg_s"] for _, pp in assumed_profiles])

                    prof0["density_mode"] = "assumed_grid"
                    prof0["assumed_ne_values_cm3"] = np.array([nn for nn, _ in assumed_profiles], dtype=float)

                    prof0["mdot_lo_msun_yr"] = np.nanmin(mdot_stack, axis=0)
                    prof0["mdot_hi_msun_yr"] = np.nanmax(mdot_stack, axis=0)
                    prof0["mdot_mid_msun_yr"] = assumed_profiles[1][1]["mdot_msun_yr"] if len(assumed_profiles) >= 2 else assumed_profiles[0][1]["mdot_msun_yr"]

                    prof0["mass_lo_msun"] = np.nanmin(mass_stack, axis=0)
                    prof0["mass_hi_msun"] = np.nanmax(mass_stack, axis=0)

                    prof0["pdot_lo_dyne"] = np.nanmin(pdot_stack, axis=0)
                    prof0["pdot_hi_dyne"] = np.nanmax(pdot_stack, axis=0)

                    prof0["edot_lo_erg_s"] = np.nanmin(edot_stack, axis=0)
                    prof0["edot_hi_erg_s"] = np.nanmax(edot_stack, axis=0)

                    return prof0

                fit_bicone_now = (str(OUTFLOW_MASK_MODE).lower() == "bicone") or bool(OUTFLOW_DOUBLE_CONE)

                ne_plus_const = None
                ne_minus_const = None

                if NE_OUTFLOW is not None:
                    if fit_bicone_now:
                        ne_plus_const = float(NE_OUTFLOW[0])
                        ne_minus_const = float(NE_OUTFLOW[1])
                    else:
                        if OUTFLOW_AXIS_SIGN >= 0:
                            ne_plus_const = float(NE_OUTFLOW[0])
                        else:
                            ne_minus_const = float(NE_OUTFLOW[0])

                energetics_pos = _build_one_lobe_energetics(
                    obs_lobe=obs_out_pos,
                    best2_profile=out_best2_pos,
                    sign_label="+",
                    fallback_ne=ne_plus_const,
                )

                energetics_neg = _build_one_lobe_energetics(
                    obs_lobe=obs_out_neg,
                    best2_profile=out_best2_neg,
                    sign_label="-",
                    fallback_ne=ne_minus_const,
                )

                if (energetics_pos is None) and (energetics_neg is None):
                    logger.warning(
                        "compute_energetics=True but no valid outflow energetics profile could be built."
                    )
                else:
                    energetics_plot_path = output_dir / "018_outflow_mdot_profile.png"
                    _plot_outflow_energetics_profile(
                        output_path=energetics_plot_path,
                        scale_kpc_per_arcsec=scale,
                        radius_range_model_out=radius_range_model_out,
                        pos_profile=energetics_pos,
                        neg_profile=energetics_neg,
                        show_plots=cfg.output.show_plots,
                    )

                    if SAVE_ENERGETICS_TABLE:
                        try:
                            if energetics_pos is not None:
                                km.save_energetics_table_fits(
                                    energetics_pos,
                                    output_dir / "outflow_energetics_pos.fits"
                                )
                            if energetics_neg is not None:
                                km.save_energetics_table_fits(
                                    energetics_neg,
                                    output_dir / "outflow_energetics_neg.fits"
                                )
                        except Exception as e:
                            logger.warning("Failed to save energetics FITS tables: %r", e)

                    if energetics_pos is not None:
                        mdot_tot = np.nansum(np.asarray(energetics_pos.get("mdot_msun_yr", np.nan), dtype=float))
                        pdot_tot = np.nansum(np.asarray(energetics_pos.get("pdot_dyne", np.nan), dtype=float))
                        edot_tot = np.nansum(np.asarray(energetics_pos.get("edot_erg_s", np.nan), dtype=float))

                        logger.info(
                            "Energetics (+): line=%s, density_mode=%s, "
                            "total_mdot=%.3e Msun/yr, total_pdot=%.3e dyne, total_edot=%.3e erg/s",
                            line_id,
                            energetics_pos.get("density_mode", "unknown"),
                            mdot_tot,
                            pdot_tot,
                            edot_tot,
                        )



                    if energetics_neg is not None:
                        mdot_tot = np.nansum(np.asarray(energetics_neg.get("mdot_msun_yr", np.nan), dtype=float))
                        pdot_tot = np.nansum(np.asarray(energetics_neg.get("pdot_dyne", np.nan), dtype=float))
                        edot_tot = np.nansum(np.asarray(energetics_neg.get("edot_erg_s", np.nan), dtype=float))

                        logger.info(
                            "Energetics (-): line=%s, density_mode=%s, "
                            "total_mdot=%.3e Msun/yr, total_pdot=%.3e dyne, total_edot=%.3e erg/s",
                            line_id,
                            energetics_neg.get("density_mode", "unknown"),
                            mdot_tot,
                            pdot_tot,
                            edot_tot,
                        )

                    if (energetics_pos is not None) or (energetics_neg is not None):

                        mdot_global = 0.0
                        pdot_global = 0.0
                        edot_global = 0.0

                        if energetics_pos is not None:
                            mdot_global += np.nansum(np.asarray(energetics_pos["mdot_msun_yr"], dtype=float))
                            pdot_global += np.nansum(np.asarray(energetics_pos["pdot_dyne"], dtype=float))
                            edot_global += np.nansum(np.asarray(energetics_pos["edot_erg_s"], dtype=float))

                        if energetics_neg is not None:
                            mdot_global += np.nansum(np.asarray(energetics_neg["mdot_msun_yr"], dtype=float))
                            pdot_global += np.nansum(np.asarray(energetics_neg["pdot_dyne"], dtype=float))
                            edot_global += np.nansum(np.asarray(energetics_neg["edot_erg_s"], dtype=float))

                        logger.info(
                            "Total outflow energetics: "
                            "Mdot=%.3e Msun/yr, Pdot=%.3e dyne, Edot=%.3e erg/s",
                            mdot_global,
                            pdot_global,
                            edot_global,
                        )

    # ============================================================
    # Escape-velocity diagnostic: v_out / v_esc
    # Only valid in disk_then_outflow mode
    # ============================================================
    escape_fraction_plot_path = None
    escape_fraction_table_path = None
    escape_fraction_singlecone_value = None
    escape_fraction_singlecone_err_value = None

    escape_pos = None
    escape_neg = None
    out_avg_prof = None

    v_c_outer = np.nan
    v_c_outer_err = np.nan
    vcirc_meta = None

    if COMPUTE_ESCAPE_FRACTION:


        if FIT_COMPONENT_MODE != "disk_then_outflow":
            logger.warning(
                "compute_escape_fraction=True but FIT_COMPONENT_MODE=%s. "
                "This diagnostic is only available in disk_then_outflow mode.",
                FIT_COMPONENT_MODE,
            )
        elif best_disc_profile is None:
            logger.warning(
                "compute_escape_fraction=True but disc profile is not available."
            )
        else:
            disc_prof = km._extract_radial_profile(
                best_profile=best_disc_profile,
                rin_pix=rin_pix_disc,
                rout_pix=rout_pix_disc,
                n_shells=num_shells_disc,
                arcsec_per_pix=arcsec_per_pix,
            )

            v_c_outer, v_c_outer_err, vcirc_meta = km._estimate_outer_vcirc(
                disc_prof,
                method="flat_plateau",
                outer_fraction=0.3,
                min_outer_points=2,
                flat_slope_frac=0.2,
                min_flat_points=2,
            )

            #logger.info(
            #    "Outer vcirc estimator details: %s",
            #    vcirc_meta
            #)

            rc_rising = km._is_rotation_curve_still_rising(disc_prof, outer_fraction=0.3)
            if rc_rising is True:
                logger.warning(
                    "Outer disc rotation curve is still rising in the observed range. "
                    "Adopted v_c_outer may underestimate the true asymptotic circular velocity; "
                    "therefore v_esc may be underestimated and v_out/v_esc may be overestimated."
                )

            logger.info(
                "Outer circular velocity adopted: v_c_outer = %.2f +/- %.2f km/s "
                "(method=%s, n_used=%d, r_range=[%.2f, %.2f] arcsec)",
                v_c_outer,
                v_c_outer_err,
                vcirc_meta["method"],
                vcirc_meta["n_used"],
                vcirc_meta.get("r_min_used_arcsec", np.nan),
                vcirc_meta.get("r_max_used_arcsec", np.nan),
            )

            if not np.isfinite(v_c_outer) or (v_c_outer <= 0):
                logger.warning(
                    "Could not determine a valid outer circular velocity; "
                    "skipping escape-velocity diagnostic."
                )
            else:
                f_eta_low = float(km._escape_factor_from_eta(np.array([ESCAPE_ETA_LOW]))[0])
                f_eta_fid = float(km._escape_factor_from_eta(np.array([ESCAPE_ETA_FID]))[0])
                f_eta_high = float(km._escape_factor_from_eta(np.array([ESCAPE_ETA_HIGH]))[0])

                logger.info(
                    "Escape-speed multipliers relative to v_c_outer: "
                    "r_{max}=10 -> %.3f, r_{max}=30 -> %.3f, r_{max}=100 -> %.3f",
                    f_eta_low, f_eta_fid, f_eta_high
                )

                out_pos_prof = None
                out_neg_prof = None
                out_avg_prof = None

                if best_out_pos_profile is not None:
                    out_pos_prof = km._extract_radial_profile(
                            best_profile=best_out_pos_profile,
                            rin_pix=rin_pix_out,
                            rout_pix=rout_pix_out,
                            n_shells=num_shells_out,
                            arcsec_per_pix=arcsec_per_pix,
                        )
    
                if best_out_neg_profile is not None:
                    out_neg_prof = km._extract_radial_profile(
                            best_profile=best_out_neg_profile,
                            rin_pix=rin_pix_out,
                            rout_pix=rout_pix_out,
                            n_shells=num_shells_out,
                            arcsec_per_pix=arcsec_per_pix,
                        )

            # --------------------------------------------------------
            # Single-shell outflow case: compute one value, but still
            # create and save a one-point plot/FITS table.
            # --------------------------------------------------------
            if int(num_shells_out) == 1:
                if out_pos_prof is not None:

                    ratio, ratio_err, v_esc_arr = km._ratio_to_escape_and_uncertainty(
                        np.array([out_pos_prof["v"][0]]),
                        np.array([out_pos_prof["v_err"][0]]),
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_FID,
                    )

                    ratio_10, _, _ = km._ratio_to_escape_and_uncertainty(
                        np.array([out_pos_prof["v"][0]]),
                        np.array([out_pos_prof["v_err"][0]]),
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_LOW,
                    )

                    ratio_100, _, _ = km._ratio_to_escape_and_uncertainty(
                        np.array([out_pos_prof["v"][0]]),
                        np.array([out_pos_prof["v_err"][0]]),
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_HIGH,
                    )


                    logger.info(
                        "Escape diagnostic (single cone, +): v_out/v_esc = %.3f ± %.3f "
                        "[eta=%.0f, v_esc=%.1f km/s]",
                        float(ratio[0]),
                        float(ratio_err[0]),
                        ESCAPE_ETA_FID,
                        float(v_esc_arr[0]),
                    )

                    escape_fraction_singlecone_value = float(ratio[0])
                    escape_fraction_singlecone_err_value = float(ratio_err[0])

                    escape_pos = {
                        "r_arcsec": np.asarray(out_pos_prof["r_arcsec"], dtype=float),
                        "xerr_arcsec": np.asarray(out_pos_prof["xerr_arcsec"], dtype=float),
                        "ratio": np.asarray(ratio, dtype=float),
                        "ratio_err": np.asarray(ratio_err, dtype=float),
                        "ratio_loweta": np.asarray(ratio_10, dtype=float),
                        "ratio_higheta": np.asarray(ratio_100, dtype=float),
                        "v_esc": np.asarray(v_esc_arr, dtype=float),
                        "eta": np.full_like(
                            np.asarray(out_pos_prof["r_arcsec"], dtype=float),
                            ESCAPE_ETA_FID,
                            dtype=float,
                        ),
                    }

                elif out_neg_prof is not None:

                    ratio, ratio_err, v_esc_arr = km._ratio_to_escape_and_uncertainty(
                        np.array([out_neg_prof["v"][0]]),
                        np.array([out_neg_prof["v_err"][0]]),
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_FID,
                    )
                    ratio_10, _, _ = km._ratio_to_escape_and_uncertainty(
                        np.array([out_neg_prof["v"][0]]),
                        np.array([out_neg_prof["v_err"][0]]),
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_LOW,
                    )

                    ratio_100, _, _ = km._ratio_to_escape_and_uncertainty(
                        np.array([out_neg_prof["v"][0]]),
                        np.array([out_neg_prof["v_err"][0]]),
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_HIGH,
                    )


                    logger.info(
                        "Escape diagnostic (single cone, -): v_out/v_esc = %.3f ± %.3f "
                        "[eta=%.0f, v_esc=%.1f km/s]",
                        float(ratio[0]),
                        float(ratio_err[0]),
                        ESCAPE_ETA_FID,
                        float(v_esc_arr[0]),
                    )

                    escape_fraction_singlecone_value = float(ratio[0])
                    escape_fraction_singlecone_err_value = float(ratio_err[0])

                    escape_neg = {
                        "r_arcsec": np.asarray(out_neg_prof["r_arcsec"], dtype=float),
                        "xerr_arcsec": np.asarray(out_neg_prof["xerr_arcsec"], dtype=float),
                        "ratio": np.asarray(ratio, dtype=float),
                        "ratio_err": np.asarray(ratio_err, dtype=float),
                        "ratio_loweta": np.asarray(ratio_10, dtype=float),
                        "ratio_higheta": np.asarray(ratio_100, dtype=float),
                        "v_esc": np.asarray(v_esc_arr, dtype=float),
                        "eta": np.full_like(
                            np.asarray(out_neg_prof["r_arcsec"], dtype=float),
                            ESCAPE_ETA_FID,
                            dtype=float,
                        ),
                    }

                else:
                    logger.warning(
                        "compute_escape_fraction=True but no valid single-shell outflow profile was available."
                    )

                # Save a one-point plot in the single-shell case
                if (escape_pos is not None) or (escape_neg is not None):
                    escape_fraction_plot_path = output_dir / "017_escape_fraction_profile.png"

                    _plot_escape_fraction_profile(
                        output_path=escape_fraction_plot_path,
                        scale_kpc_per_arcsec=scale,
                        radius_range_model_disc=radius_range_model_disc,
                        radius_range_model_out=radius_range_model_out,
                        disc_profile=disc_prof,
                        out_pos_profile=escape_pos,
                        out_neg_profile=escape_neg,
                        out_avg_profile=None,
                        show_plots=cfg.output.show_plots,
                    )

                # Save optional FITS table also in the single-shell case
                if SAVE_ESCAPE_FRACTION_TABLE and ((escape_pos is not None) or (escape_neg is not None)):
                    try:
                        escape_fraction_table_path = output_dir / "escape_fraction_profile.fits"
                        km._save_escape_fraction_table_fits(
                            {
                                "ESCAPE_POS": escape_pos,
                                "ESCAPE_NEG": escape_neg,
                                "ESCAPE_AVG": None,
                            },
                            escape_fraction_table_path,
                        )
                    except Exception as e:
                        logger.warning("Failed to save escape fraction FITS table: %r", e)

            # --------------------------------------------------------
            # Multi-shell / general case
            # --------------------------------------------------------
            else:
                def _build_escape_profile(out_prof, eta):
                    if out_prof is None or disc_prof is None:
                        return None

                    ratio, ratio_err, v_esc = km._ratio_to_escape_and_uncertainty(
                        out_prof["v"],
                        out_prof["v_err"],
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=eta,
                    )

                    return {
                        "r_arcsec": np.asarray(out_prof["r_arcsec"], dtype=float),
                        "xerr_arcsec": np.asarray(out_prof["xerr_arcsec"], dtype=float),
                        "ratio": np.asarray(ratio, dtype=float),
                        "ratio_err": np.asarray(ratio_err, dtype=float),
                        "v_esc": np.asarray(v_esc, dtype=float),
                        "eta": np.full_like(out_prof["r_arcsec"], float(eta), dtype=float),
                    }

                escape_pos_10 = _build_escape_profile(out_pos_prof, eta=10.0)
                escape_pos_30 = _build_escape_profile(out_pos_prof, eta=30.0)
                escape_pos_100 = _build_escape_profile(out_pos_prof, eta=100.0)

                escape_neg_10 = _build_escape_profile(out_neg_prof, eta=10.0)
                escape_neg_30 = _build_escape_profile(out_neg_prof, eta=30.0)
                escape_neg_100 = _build_escape_profile(out_neg_prof, eta=100.0)

                # Use eta=30 as the fiducial profile for plotting/table output
                escape_pos = escape_pos_30
                escape_neg = escape_neg_30

                if escape_pos is not None:
                    escape_pos["ratio_loweta"] = np.asarray(escape_pos_10["ratio"], dtype=float) if escape_pos_10 is not None else np.full_like(escape_pos["ratio"], np.nan, dtype=float)
                    escape_pos["ratio_higheta"] = np.asarray(escape_pos_100["ratio"], dtype=float) if escape_pos_100 is not None else np.full_like(escape_pos["ratio"], np.nan, dtype=float)

                if escape_neg is not None:
                    escape_neg["ratio_loweta"] = np.asarray(escape_neg_10["ratio"], dtype=float) if escape_neg_10 is not None else np.full_like(escape_neg["ratio"], np.nan, dtype=float)
                    escape_neg["ratio_higheta"] = np.asarray(escape_neg_100["ratio"], dtype=float) if escape_neg_100 is not None else np.full_like(escape_neg["ratio"], np.nan, dtype=float)




                if escape_pos_10 is not None and escape_pos_30 is not None and escape_pos_100 is not None:
                    logger.info(
                        "Outflow (+) escape diagnostic built for r_{max} = %.0f kpc, %.0f kpc, %.0f kpc",
                        ESCAPE_ETA_LOW, ESCAPE_ETA_FID, ESCAPE_ETA_HIGH
                    )

                if escape_neg_10 is not None and escape_neg_30 is not None and escape_neg_100 is not None:
                    logger.info(
                        "Outflow (-) escape diagnostic built for r_{max} = %.0f kpc, %.0f kpc, %.0f kpc",
                        ESCAPE_ETA_LOW, ESCAPE_ETA_FID, ESCAPE_ETA_HIGH
                    )

                # Average of the two cones, shell by shell, with fiducial and halo-range envelopes
                if (out_pos_prof is not None) and (out_neg_prof is not None):
                    v_avg = 0.5 * (out_pos_prof["v"] + out_neg_prof["v"])
                    e_avg = 0.5 * np.sqrt(out_pos_prof["v_err"]**2 + out_neg_prof["v_err"]**2)

                    ratio_avg_10, _, _ = km._ratio_to_escape_and_uncertainty(
                        v_avg,
                        e_avg,
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_LOW,
                    )

                    ratio_avg_30, ratio_avg_err, v_esc_avg = km._ratio_to_escape_and_uncertainty(
                        v_avg,
                        e_avg,
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_FID,
                    )

                    ratio_avg_100, _, _ = km._ratio_to_escape_and_uncertainty(
                        v_avg,
                        e_avg,
                        v_c_outer=v_c_outer,
                        e_c_outer=v_c_outer_err,
                        eta=ESCAPE_ETA_HIGH,
                    )

                    out_avg_prof = {
                        "r_arcsec": np.asarray(out_pos_prof["r_arcsec"], dtype=float),
                        "xerr_arcsec": np.asarray(out_pos_prof["xerr_arcsec"], dtype=float),
                        "ratio": np.asarray(ratio_avg_30, dtype=float),
                        "ratio_err": np.asarray(ratio_avg_err, dtype=float),
                        "v_esc": np.asarray(v_esc_avg, dtype=float),
                        "eta": np.full_like(np.asarray(out_pos_prof["r_arcsec"], dtype=float), ESCAPE_ETA_FID, dtype=float),
                        "ratio_loweta": np.asarray(ratio_avg_10, dtype=float),
                        "ratio_higheta": np.asarray(ratio_avg_100, dtype=float),
                    }
                else:
                    out_avg_prof = None

                if (escape_pos is None) and (escape_neg is None):
                    logger.warning(
                        "compute_escape_fraction=True but no valid outflow escape-fraction profile could be built."
                    )
                else:
                    escape_fraction_plot_path = output_dir / "017_escape_fraction_profile.png"

                    if (escape_pos is not None) and (escape_neg is None):
                        _plot_escape_fraction_profile(
                            output_path=escape_fraction_plot_path,
                            scale_kpc_per_arcsec=scale,
                            radius_range_model_disc=radius_range_model_disc,
                            radius_range_model_out=radius_range_model_out,
                            disc_profile=disc_prof,
                            out_pos_profile=escape_pos,
                            out_neg_profile=None,
                            out_avg_profile=None,
                            show_plots=cfg.output.show_plots,
                        )

                    elif (escape_neg is not None) and (escape_pos is None):
                        _plot_escape_fraction_profile(
                            output_path=escape_fraction_plot_path,
                            scale_kpc_per_arcsec=scale,
                            radius_range_model_disc=radius_range_model_disc,
                            radius_range_model_out=radius_range_model_out,
                            disc_profile=disc_prof,
                            out_pos_profile=None,
                            out_neg_profile=escape_neg,
                            out_avg_profile=None,
                            show_plots=cfg.output.show_plots,
                        )

                    else:
                        _plot_escape_fraction_profile(
                            output_path=escape_fraction_plot_path,
                            scale_kpc_per_arcsec=scale,
                            radius_range_model_disc=radius_range_model_disc,
                            radius_range_model_out=radius_range_model_out,
                            disc_profile=disc_prof,
                            out_pos_profile=escape_pos,
                            out_neg_profile=escape_neg,
                            out_avg_profile=out_avg_prof,
                            show_plots=cfg.output.show_plots,
                        )

                    if SAVE_ESCAPE_FRACTION_TABLE:
                        try:
                            escape_fraction_table_path = output_dir / "escape_fraction_profile.fits"
                            km._save_escape_fraction_table_fits(
                                {
                                    "ESCAPE_POS": escape_pos,
                                    "ESCAPE_NEG": escape_neg,
                                    "ESCAPE_AVG": out_avg_prof,
                                },
                                escape_fraction_table_path,
                            )
                        except Exception as e:
                            logger.warning("Failed to save escape fraction FITS table: %r", e)



    # ============================================================
    # SAVE ALL OUTPUTS
    # ============================================================
    if SAVE_ALL_OUTPUTS:
        if final_model is not None:
            try:
                cube_out = output_dir / final_model_name
                km._save_model_cube_fits(final_model, obs, cube_out)
                logger.info("Saved best-fit weighted model cube to %s", cube_out)
            except Exception as e:
                logger.warning("Failed to save best-fit model cube: %r", e)

            try:
                maps_out = output_dir / "bestfit_moment_maps.fits"
                km._save_moment_maps_fits(obs, final_model, maps_out)
                logger.info("Saved data/model/residual moment maps to %s", maps_out)
            except Exception as e:
                logger.warning("Failed to save moment maps FITS: %r", e)
        else:
            logger.warning("save_all_outputs=True but no final model was available to save.")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    fit_bicone = (str(OUTFLOW_MASK_MODE).lower() == "bicone") or bool(OUTFLOW_DOUBLE_CONE)
    disc_present = FIT_COMPONENT_MODE in ("disk", "disk_then_outflow")
    outflow_present = FIT_COMPONENT_MODE in ("outflow", "disk_then_outflow")
    is_single_outflow = outflow_present and (not fit_bicone)
    is_bicone_outflow = outflow_present and fit_bicone

    summary["fit_component_mode"] = FIT_COMPONENT_MODE
    summary["check_masking_before_fitting"] = bool(cfg.advanced.check_masking_before_fitting)

    summary["num_shells_disc"] = int(num_shells_disc)
    summary["num_shells_out"] = int(num_shells_out)

    # Record the selected outflow topology whenever outflow is part of the run
    if outflow_present:
        summary["outflow_topology"] = "bicone" if fit_bicone else "single"
        summary["outflow_axis_sign"] = int(OUTFLOW_AXIS_SIGN)
        summary["outflow_pa_deg"] = float(OUTFLOW_PA_DEG)
        summary["outflow_opening_deg"] = float(OUTFLOW_OPENING_DEG)
    else:
        summary["outflow_topology"] = None
        summary["outflow_axis_sign"] = None
        summary["outflow_pa_deg"] = None
        summary["outflow_opening_deg"] = None

    # ----------------------------
    # Disc PA
    # ----------------------------
    if gamma_disc is not None:
        summary["disc_pa_deg"] = float(gamma_disc)
        summary["disc_pa_unc_deg"] = (
            float(gamma_disc_unc) if gamma_disc_unc is not None else None
        )
        summary["disc_pa_mode"] = "fixed" if disc_cfg.pa_deg is not None else "estimated"
    else:
        summary["disc_pa_deg"] = None
        summary["disc_pa_unc_deg"] = None
        summary["disc_pa_mode"] = None

    # ----------------------------
    # Disc fit
    # ----------------------------
    if disc_present and (disc_fit is not None):
        disc_best_info = km._extract_best_fit_with_uncertainties(disc_fit)
        summary["disc_best_beta_deg"] = disc_best_info["beta_best"]
        summary["disc_best_beta_err_deg"] = disc_best_info["beta_err"]
        summary["disc_fit_mode"] = DISC_FIT_MODE
        summary["disc_best_param"] = disc_best_info["v_best"]
        summary["disc_best_param_err"] = disc_best_info["v_err"]

    else:
        summary["disc_best_beta_deg"] = None
        summary["disc_best_beta_err_deg"] = None
        summary["disc_fit_mode"] = DISC_FIT_MODE
        summary["disc_best_param"] = None
        summary["disc_best_param_err"] = None

    # ----------------------------
    # Single-cone outflow
    # ----------------------------
    if is_single_outflow:
        summary["outflow_lobe"] = None
        summary["outflow_best_beta_deg"] = None
        summary["outflow_best_beta_err_deg"] = None
        summary["outflow_best_v_kms"] = None
        summary["outflow_best_v_err_kms"] = None

        if outflow_fit_pos is not None:
            out_best_info = km._extract_best_fit_with_uncertainties(outflow_fit_pos)
            summary["outflow_best_beta_deg"] = out_best_info["beta_best"]
            summary["outflow_best_beta_err_deg"] = out_best_info["beta_err"]
            summary["outflow_best_v_kms"] = out_best_info["v_best"]
            summary["outflow_best_v_err_kms"] = out_best_info["v_err"]
            summary["outflow_lobe"] = "positive"

        elif outflow_fit_neg is not None:
            out_best_info = km._extract_best_fit_with_uncertainties(outflow_fit_neg)
            summary["outflow_best_beta_deg"] = out_best_info["beta_best"]
            summary["outflow_best_beta_err_deg"] = out_best_info["beta_err"]
            summary["outflow_best_v_kms"] = out_best_info["v_best"]
            summary["outflow_best_v_err_kms"] = out_best_info["v_err"]
            summary["outflow_lobe"] = "negative"

        # In single-cone mode, keep lobe-specific bicone fields explicitly null
        summary["outflow_pos_best_beta_deg"] = None
        summary["outflow_pos_best_beta_err_deg"] = None
        summary["outflow_pos_best_v_kms"] = None
        summary["outflow_pos_best_v_err_kms"] = None

        summary["outflow_neg_best_beta_deg"] = None
        summary["outflow_neg_best_beta_err_deg"] = None
        summary["outflow_neg_best_v_kms"] = None
        summary["outflow_neg_best_v_err_kms"] = None

    # ----------------------------
    # Bicone outflow
    # ----------------------------
    elif is_bicone_outflow:
        # No compact single-outflow fields in bicone mode
        summary["outflow_lobe"] = None
        summary["outflow_best_beta_deg"] = None
        summary["outflow_best_beta_err_deg"] = None
        summary["outflow_best_v_kms"] = None
        summary["outflow_best_v_err_kms"] = None

        if outflow_fit_pos is not None:
            out_pos_info = km._extract_best_fit_with_uncertainties(outflow_fit_pos)
            summary["outflow_pos_best_beta_deg"] = out_pos_info["beta_best"]
            summary["outflow_pos_best_beta_err_deg"] = out_pos_info["beta_err"]
            summary["outflow_pos_best_v_kms"] = out_pos_info["v_best"]
            summary["outflow_pos_best_v_err_kms"] = out_pos_info["v_err"]
        else:
            summary["outflow_pos_best_beta_deg"] = None
            summary["outflow_pos_best_beta_err_deg"] = None
            summary["outflow_pos_best_v_kms"] = None
            summary["outflow_pos_best_v_err_kms"] = None

        if outflow_fit_neg is not None:
            out_neg_info = km._extract_best_fit_with_uncertainties(outflow_fit_neg)
            summary["outflow_neg_best_beta_deg"] = out_neg_info["beta_best"]
            summary["outflow_neg_best_beta_err_deg"] = out_neg_info["beta_err"]
            summary["outflow_neg_best_v_kms"] = out_neg_info["v_best"]
            summary["outflow_neg_best_v_err_kms"] = out_neg_info["v_err"]
        else:
            summary["outflow_neg_best_beta_deg"] = None
            summary["outflow_neg_best_beta_err_deg"] = None
            summary["outflow_neg_best_v_kms"] = None
            summary["outflow_neg_best_v_err_kms"] = None

    # ----------------------------
    # No outflow 
    # ----------------------------
    else:
        summary["outflow_lobe"] = None
        summary["outflow_best_beta_deg"] = None
        summary["outflow_best_beta_err_deg"] = None
        summary["outflow_best_v_kms"] = None
        summary["outflow_best_v_err_kms"] = None

        summary["outflow_pos_best_beta_deg"] = None
        summary["outflow_pos_best_beta_err_deg"] = None
        summary["outflow_pos_best_v_kms"] = None
        summary["outflow_pos_best_v_err_kms"] = None

        summary["outflow_neg_best_beta_deg"] = None
        summary["outflow_neg_best_beta_err_deg"] = None
        summary["outflow_neg_best_v_kms"] = None
        summary["outflow_neg_best_v_err_kms"] = None



    # ----------------------------
    # Escape-velocity diagnostic summary
    # Only meaningful for disk_then_outflow mode
    # ----------------------------

    summary["escape_vcirc_outer_kms"] = None
    summary["escape_vcirc_outer_err_kms"] = None
    summary["escape_vcirc_outer_method"] = None
    summary["escape_vcirc_outer_rising_flag"] = None
    summary["escape_eta_low"] = ESCAPE_ETA_LOW
    summary["escape_eta_fid"] = ESCAPE_ETA_FID
    summary["escape_eta_high"] = ESCAPE_ETA_HIGH


    summary["compute_escape_fraction"] = bool(COMPUTE_ESCAPE_FRACTION)
    summary["escape_fraction_singlecone"] = None
    summary["escape_fraction_singlecone_err"] = None
    summary["escape_fraction_plot"] = None
    summary["escape_fraction_table_fits"] = None

    summary["escape_fraction_pos_available"] = None
    summary["escape_fraction_neg_available"] = None
    summary["escape_fraction_avg_available"] = None

    summary["compute_energetics"] = bool(COMPUTE_ENERGETICS)
    summary["energetics_plot_path"] = str(energetics_plot_path) if energetics_plot_path is not None else None
    summary["energetics_pos_density_mode"] = energetics_pos.get("density_mode", None) if energetics_pos is not None else None
    summary["energetics_neg_density_mode"] = energetics_neg.get("density_mode", None) if energetics_neg is not None else None


    summary["energetics_pos_total_mdot_msun_yr"] = (
    float(np.nansum(energetics_pos["mdot_msun_yr"])) if energetics_pos is not None else None
    )

    summary["energetics_pos_total_pdot_dyne"] = (
    float(np.nansum(energetics_pos["pdot_dyne"])) if energetics_pos is not None else None
    )

    summary["energetics_pos_total_edot_erg_s"] = (
    float(np.nansum(energetics_pos["edot_erg_s"])) if energetics_pos is not None else None
    )

    summary["energetics_neg_total_mdot_msun_yr"] = (
    float(np.nansum(energetics_neg["mdot_msun_yr"])) if energetics_neg is not None else None
    )

    summary["energetics_neg_total_pdot_dyne"] = (
    float(np.nansum(energetics_neg["pdot_dyne"])) if energetics_neg is not None else None
    )

    summary["energetics_neg_total_edot_erg_s"] = (
    float(np.nansum(energetics_neg["edot_erg_s"])) if energetics_neg is not None else None
    )


    summary["energetics_total_mdot_msun_yr"] = (
        float(mdot_global) if ("mdot_global" in locals()) else None
    )
    summary["energetics_total_pdot_dyne"] = (
        float(pdot_global) if ("pdot_global" in locals()) else None
    )
    summary["energetics_total_edot_erg_s"] = (
        float(edot_global) if ("edot_global" in locals()) else None
    )




    if COMPUTE_ESCAPE_FRACTION and FIT_COMPONENT_MODE == "disk_then_outflow":

        if "v_c_outer" in locals() and np.isfinite(v_c_outer):
            summary["escape_vcirc_outer_kms"] = float(v_c_outer)

        if "v_c_outer_err" in locals() and np.isfinite(v_c_outer_err):
            summary["escape_vcirc_outer_err_kms"] = float(v_c_outer_err)

        if "vcirc_meta" in locals() and isinstance(vcirc_meta, dict):
            summary["escape_vcirc_outer_method"] = vcirc_meta.get("method", None)

        if "rc_rising" in locals():
            summary["escape_vcirc_outer_rising_flag"] = None if rc_rising is None else bool(rc_rising)

        # These variables should have been set in the escape-fraction block
        # If they do not exist, keep the fields as None
        if "escape_fraction_plot_path" in locals() and escape_fraction_plot_path is not None:
            summary["escape_fraction_plot"] = str(Path(escape_fraction_plot_path).name)

        if "escape_fraction_table_path" in locals() and escape_fraction_table_path is not None:
            summary["escape_fraction_table_fits"] = str(Path(escape_fraction_table_path).name)

        if "escape_pos" in locals():
            summary["escape_fraction_pos_available"] = escape_pos is not None
        else:
            summary["escape_fraction_pos_available"] = False

        if "escape_neg" in locals():
            summary["escape_fraction_neg_available"] = escape_neg is not None
        else:
            summary["escape_fraction_neg_available"] = False

        if "out_avg_prof" in locals():
            summary["escape_fraction_avg_available"] = out_avg_prof is not None
        else:
            summary["escape_fraction_avg_available"] = False

        # Single-shell/single-cone compact value
        if "escape_fraction_singlecone_value" in locals() and escape_fraction_singlecone_value is not None:
            summary["escape_fraction_singlecone"] = float(escape_fraction_singlecone_value)

        if "escape_fraction_singlecone_err_value" in locals() and escape_fraction_singlecone_err_value is not None:
            summary["escape_fraction_singlecone_err"] = float(escape_fraction_singlecone_err_value)


    # Save summary only once, at the very end
    if cfg.output.save_summary_json:
        _save_summary(summary, output_dir)

    return summary





