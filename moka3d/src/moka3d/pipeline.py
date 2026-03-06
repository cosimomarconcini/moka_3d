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

import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18 as cosmo
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
from skimage.transform import downscale_local_mean
from matplotlib.ticker import AutoMinorLocator

from . import moka3d_source as km
from .plotting import finalize_figure



logger = logging.getLogger(__name__)


            
def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "moka3d.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def _save_summary(summary: dict, output_dir: Path) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _disc_zeta_range(cfg):
    if cfg.advanced.disc_zeta_range_mode == "auto_from_psf":
        return [-cfg.processing.psf_sigma / 2.0, cfg.processing.psf_sigma / 2.0]
    raise ValueError(f"Unsupported disc_zeta_range_mode: {cfg.advanced.disc_zeta_range_mode}")


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

    if cfg.input.sn_map is not None:
        obscube = km.apply_sn_mask_to_cube(
            obscube,
            str(cfg.paths.ancillary_dir),
            cfg.input.sn_map,
            cfg.processing.sn_thresh
        )

    pixscale = km.pixel_scale_arcsec(wcs_large)
    n_spec = int(obshead.get("NAXIS3", 0))
    spec_coord, spec_unit = km.spectral_axis_from_header_general(obshead, n_spec)

    vel_kms_approx, spec_kind, line_obs = km.velocity_axis_from_spectral_coord(
        spec_coord,
        spec_unit,
        line_value=cfg.line.wavelength_line,
        line_unit=cfg.line.wavelength_line_unit,
        redshift=cfg.target.redshift,
        convention="radio",
    )

    obscube, _ = km.standardize_cube_to_spec_yx(obscube, n_spec=len(vel_kms_approx))

    with np.errstate(all="ignore"):
        floor_map = np.nanmin(obscube, axis=0)

    obscube = obscube - floor_map[None, :, :]
    obscube = np.clip(obscube, 0.0, None)

    i0 = int(np.nanargmin(np.abs(vel_kms_approx)))
    vel_kms = km.velocity_axis_rezero_to_systemic(vel_kms_approx, i0)

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
    flrange = np.nanpercentile(obs.maps["flux"][pos], cfg.processing.percentile_shown_mom_maps[0])
    velrange = np.nanpercentile(obs.maps["vel"], cfg.processing.percentile_shown_mom_maps[1])
    sigrange = np.nanpercentile(obs.maps["sig"], cfg.processing.percentile_shown_mom_maps[2])

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
    )

    summary = {
        "input_cube": cfg.input.cube_file,
        "hdu_index": hdu_index,
        "spec_unit": str(spec_unit),
        "pixel_scale_arcsec": float(pixscale),
        "luminosity_distance": float(D_L),
        "angular_diameter_distance": float(D_A),
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
    logger.info("Estimated PA = %.2f +/- %.2f deg", pa_est, pa_est_unc)
    
    

    # ============================================================
    # FIT CONFIGURATION FROM YAML FILE
    # ============================================================
    
    FIT_COMPONENT_MODE = cfg.fit.component_mode
    
    USE_CRPS = cfg.advanced.use_crps
    loss = "crps" if USE_CRPS else "extreme"
    
    perc_disc = list(cfg.advanced.perc_disc)
    perc_out = list(cfg.advanced.perc_out)
    perc_weights = list(cfg.advanced.perc_weights)
    
    npt = int(cfg.advanced.npt)
    
    radius_range_model_disc = list(cfg.fit.radius_range_model_disc)
    radius_range_model_out = list(cfg.fit.radius_range_model_out)
    
    num_shells_disc = int(cfg.fit.num_shells_disc)
    num_shells_out = int(cfg.fit.num_shells_out)
    
    beta_min_d, beta_max_d, step_beta_d = map(float, cfg.fit.beta_grid_disc)
    v_min_d, v_max_d, step_v_d = map(float, cfg.fit.v_grid_disc)
    
    OUTFLOW_PA_DEG = float(cfg.fit.outflow_pa_deg)
    OUTFLOW_OPENING_DEG = float(cfg.fit.outflow_opening_deg)
    OUTFLOW_DOUBLE_CONE = bool(cfg.fit.outflow_double_cone)
    OUTFLOW_MASK_MODE = str(cfg.fit.outflow_mask_mode)
    OUTFLOW_AXIS_SIGN = int(cfg.advanced.outflow_axis_sign)
    
    beta_min_o, beta_max_o, step_beta_o = map(float, cfg.fit.beta_grid_out)
    v_min_o, v_max_o, step_v_o = map(float, cfg.fit.v_grid_out)
    
    USE_GLOBAL_BETA_DISC = bool(cfg.advanced.use_global_beta_disc)
    DISC_FIT_MODE = str(cfg.advanced.disc_fit_mode)
    disc_phi_range = cfg.advanced.disc_phi_range
    disc_zeta_range = _disc_zeta_range(cfg)
    
    USE_GLOBAL_BETA_OUT = bool(cfg.advanced.use_global_beta_out)
    
    MASK_DISK_WITH_OUTFLOW = bool(cfg.advanced.mask_disk_with_outflow)
    # MASK_MODE = str(cfg.advanced.mask_mode)
    DO_FINAL_COMBINED_MODEL_PLOT = bool(cfg.advanced.do_final_combined_model_plot)
    resid_ranges = list(cfg.advanced.resid_ranges)
    
    # Processing/runtime aliases used throughout the old script
    vel = vel_kms
    nrebin = int(cfg.processing.nrebin)
    psf_sigma = float(cfg.processing.psf_sigma)
    lsf_sigma = float(cfg.processing.lsf_sigma)
    vel_sigma = float(cfg.processing.vel_sigma)
    logradius = bool(cfg.processing.logradius)
    xrange = cfg.processing.xrange
    yrange = cfg.processing.yrange
        
    dv_chan = float(np.nanmedian(np.abs(np.diff(vel))))
    SIGMA_PERC_KMS = float(np.sqrt(lsf_sigma**2 + (0.5 * dv_chan)**2))
    
    arcsec_per_pix = pixscale * nrebin
    rin_pix_disc = int(round(radius_range_model_disc[0] / arcsec_per_pix))
    rout_pix_disc = int(round(radius_range_model_disc[1] / arcsec_per_pix))
    rin_pix_out = int(round(radius_range_model_out[0] / arcsec_per_pix))
    rout_pix_out = int(round(radius_range_model_out[1] / arcsec_per_pix))
    
    cube_range = obs.cube["range"]
    cube_nbins = obs.cube["nbins"]
    
    logger.info("Fit mode: %s", FIT_COMPONENT_MODE)
    logger.info("Disc shells: %d | Outflow shells: %d", num_shells_disc, num_shells_out)
    
    
    # fixed standard params that could be also decided by the user even if are std
    CRPS_QGRID = np.linspace(0.01, 0.99, 19)   
    disc_geometry = "cylindrical" # disc geometry is always cylindrical
    disc_theta_range = [[0, 1]]
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
    # 1) Build outflow spatial masks: lobe +, lobe -, bicone (union)
    # --------------------------
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

    def _keep_only_mask(cube, keep_mask_yx, mode="nan"):
        """Keep pixels inside keep_mask_yx; mask everything else."""
        mask_outside = ~keep_mask_yx
        return km.apply_spatial_mask_to_cube(cube, mask_outside, mode=mode)


    # ========================================
    # 2) DISC FIT
    # ========================================
    disc_fit = None
    model_disc_best = None
    obs_disc_fit = None
    disc_best2_for_plots = None
    disc_cube_for_outflow = None

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
        gamma_disc, gamma_disc_unc = km.estimate_pa_from_mom1(
            obs_disc_fit.maps['vel'],
            center_xy=origin,
            pixscale=pixscale,
            nrebin=nrebin,
            xlimshow=xrange,
            ylimshow=yrange,
            psf_sigma_arcsec=psf_sigma,
            R_data_arcsec=R_int_arcsec,
            R_data_err_arcsec=R_int_err
        )
        logger.info("DISC PA used: %.1f +/- %.1f deg", gamma_disc, gamma_disc_unc)
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
            verbose_label="DISC"
        )

        logger.info("DISC best: beta=%.1f deg, v=%.1f km/s",disc_fit["beta_best"],disc_fit["v_best"])

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

        if USE_GLOBAL_BETA_DISC:
            disc_fit["beta_best"] = float(beta_best_global_disc)

        # summarize for plots + scatter
        try:
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
        if USE_GLOBAL_BETA_DISC:
            model_disc_best = km.build_best_model_from_fit(
                beta_best=float(disc_fit["beta_best"]),
                v_best=float(disc_fit["v_best"]),
                FIT_MODE=DISC_FIT_MODE
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
            km.show_shells_overlay(
                cube_obs=obs_disc_fit.cube["data"], cube_model=model_disc_best.cube["data"],
                center_xy=origin, inc_deg=inc_for_overlay, pa_deg=float(gamma_disc),
                n_shells=num_shells_disc, r_min_pix=rin_pix_disc, r_max_pix=rout_pix_disc,
                aperture_deg=disc_aperture, double_cone=disc_double_cone,
                pixscale=pixscale, nrebin=nrebin, scale=scale,
                mask_mode="model", edges_mode="model",
                title=fr"DISC: β={disc_fit['beta_best']:.1f}°, v={disc_fit['v_best']:.0f} km s$^{{-1}}$",
                debug_intrinsic=(disc_geometry.lower() == "cylindrical"),
                xlimit=xrange, ylimit=yrange
            )
            finalize_figure(output_dir / "04_disc_shell_overlay.png", show=cfg.output.show_plots)

        except Exception as e:
            logger.warning("DISC overlay failed: %r", e)


        km.plot_residual_maps_cone(
            disc_fit["chi_squared_map"], disc_fit["beta_array"], disc_fit["v_array"],
            num_shells_disc_eff, best=disc_best2_for_plots, y_label=r"$v$ (km s$^{-1}$)"
        )
        finalize_figure(output_dir / "05_disc_residual_maps.png", show=cfg.output.show_plots)

        try:
            km.inspect_percentiles_at(
                float(disc_fit["beta_best"]), float(disc_fit["v_best"]),
                perc=perc_disc, perc_weights=perc_weights,
                sigma_perc_kms=SIGMA_PERC_KMS, loss=loss, qgrid=CRPS_QGRID
            )
            finalize_figure(output_dir / "06_disc_percentiles.png", show=cfg.output.show_plots)

        except Exception as e:
            logger.warning("DISC inspect failed: %r", e)


            
            
        # ============================================================
        # DISC-ONLY: moment maps comparison 
        # ============================================================
        if (FIT_COMPONENT_MODE == "disk") and DO_FINAL_COMBINED_MODEL_PLOT and (model_disc_best is not None):
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
        logger.info("%s best: beta=%.1f deg, v=%.1f km/s",label,fit["beta_best"],fit["v_best"])

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
        finalize_figure(output_dir / f"09_{safe_label}_residual_maps.png", show=cfg.output.show_plots)

        try:
            km.inspect_percentiles_at(
                float(fit["beta_best"]), float(fit["v_best"]),
                perc=perc_out, perc_weights=perc_weights,
                sigma_perc_kms=SIGMA_PERC_KMS, loss=loss, qgrid=CRPS_QGRID
            )
            finalize_figure(output_dir / f"10_{safe_label}_percentiles.png", show=cfg.output.show_plots)

        except Exception as e:
            logger.warning("label %r inspect %r", label, e)


        return fit, model_best, best2


    if FIT_COMPONENT_MODE in ("outflow", "disk_then_outflow"):

        if FIT_COMPONENT_MODE == "outflow":
            km.set_fit_context(disc_cube=None)
        # lobe +
        logger.info("OUTFLOW lobe +: PA=%.1f deg, opening=%.1f deg", OUTFLOW_PA_DEG,OUTFLOW_OPENING_DEG)

        outflow_fit_pos, model_outflow_pos_best, out_best2_pos = _run_single_lobe_outflow_fit(
            obs_lobe=obs_out_pos,
            pa_deg=OUTFLOW_PA_DEG,
            label="OUTFLOW (+)"
        )

        # lobe -
        gamma_neg = (OUTFLOW_PA_DEG + 180.0) % 360.0
        logger.info("OUTFLOW lobe -: PA=%.1f deg, opening=%.1f deg", gamma_neg,OUTFLOW_OPENING_DEG)

        outflow_fit_neg, model_outflow_neg_best, out_best2_neg = _run_single_lobe_outflow_fit(
            obs_lobe=obs_out_neg,
            pa_deg=gamma_neg,
            label="OUTFLOW (-)"
        )

    # ============================================================
    # OUTFLOW-ONLY: build total outflow model 
    # ============================================================
    if (FIT_COMPONENT_MODE == "outflow") and DO_FINAL_COMBINED_MODEL_PLOT:

        
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
                    npt=int(npt)*10,
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
                    npt_total=int(npt)*10,
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
    if DO_FINAL_COMBINED_MODEL_PLOT and (model_disc_best is not None) and (model_outflow_pos_best is not None) and (model_outflow_neg_best is not None):

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
        radius_shells_disc = km._as_shell_ranges(radius_range_model_disc, num_shells_disc_eff)
        if USE_GLOBAL_BETA_DISC:
            beta_arr_disc = [beta_disc_best] * num_shells_disc_eff
            v_arr_disc    = [v_disc_best]    * num_shells_disc_eff
        else:
            beta_arr_disc = list(np.asarray(disc_best2_for_plots["beta"], float))
            v_arr_disc    = list(np.asarray(disc_best2_for_plots["v"], float))

        m_final = km._make_multishell_component(
            npt_total=int(npt)*10,
            n_shells=num_shells_disc_eff,
            geometry="cylindrical",
            radius_range_shells=radius_shells_disc,
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
                    npt=int(npt)*10,
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
                    npt_total=int(npt)*10,
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
        best_disc_profile = disc_fit.get("best", None)
        km._plot_v_profile(
            best_disc_profile,
            n_shells=num_shells_disc,
            title="Disc velocity profile",
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
        finalize_figure(output_dir / "013_out+_vel_profiles.png", show=cfg.output.show_plots)


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
        finalize_figure(output_dir / "99_out-_vel_profiles.png", show=cfg.output.show_plots)


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

        ax_top = ax.twiny()
        ax_top.tick_params(axis="both", labelsize=12)
        ax_top.set_xlim(ax.get_xlim())

        tick_arc = ax.get_xticks()
        tick_arc = tick_arc[tick_arc >= 0]
        ax.set_xticks(tick_arc)
        ax_top.set_xticks(tick_arc)
        tick_show = tick_arc * scale
        ax_top.set_xticklabels([f"{round(t,1):.1f}" for t in tick_show])
        ax_top.set_xlabel("Radius [kpc]", fontsize=14)
        ax.set_xlim(xmin, xmax)

        plt.tight_layout()
        finalize_figure(output_dir / "99_vel_profiles.png", show=cfg.output.show_plots)


    summary["fit_component_mode"] = FIT_COMPONENT_MODE

    if disc_fit is not None:
        summary["disc_best_beta_deg"] = float(disc_fit["beta_best"])
        summary["disc_best_v_kms"] = float(disc_fit["v_best"])

    if outflow_fit_pos is not None:
        summary["outflow_pos_best_beta_deg"] = float(outflow_fit_pos["beta_best"])
        summary["outflow_pos_best_v_kms"] = float(outflow_fit_pos["v_best"])

    if outflow_fit_neg is not None:
        summary["outflow_neg_best_beta_deg"] = float(outflow_fit_neg["beta_best"])
        summary["outflow_neg_best_v_kms"] = float(outflow_fit_neg["v_best"])

    if cfg.output.save_summary_json:
        _save_summary(summary, output_dir)
        
        
    return summary










