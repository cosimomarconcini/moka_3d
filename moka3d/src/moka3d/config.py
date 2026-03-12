#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:37:15 2026

@author: cosimo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal, Any
import yaml



FitMode = Literal["disk", "outflow", "disk_then_outflow"]
MaskMode = Literal["single", "bicone"]
CenterMode = Optional[Literal["flux", "kinematic"]]
DiscModelMode = Literal["independent", "disk_kepler", "NSC", "Plummer", "disk_arctan"]


@dataclass
class DisplayRangeConfig:
    mode: Literal["percentile", "fixed"] = "percentile"
    values: List[float] = field(default_factory=lambda: [1.0, 99.0])

@dataclass
class PathsConfig:
    data_dir: Path
    ancillary_dir: Path
    output_dir: Path


@dataclass
class InputConfig:
    cube_file: str
    sn_map: Optional[str] = None
    save_all_outputs: bool = False


@dataclass
class TargetConfig:
    agn_ra: Optional[Any] = None
    agn_dec: Optional[Any] = None
    center_mode: CenterMode = None
    center_xy_manual: Optional[List[int]] = None
    redshift: float = 0.0


@dataclass
class LineConfig:
    wavelength_line: float
    wavelength_line_unit: str = "Angstrom"


@dataclass
class ProcessingConfig:
    sn_thresh: Optional[float] = 3.0
    nrebin: int = 1
    xrange: Optional[List[float]] = None
    yrange: Optional[List[float]] = None
    pixel_scale_arcsec_manual: Optional[float] = None
    psf_sigma: float = 1.0
    lsf_sigma: float = 72.0
    vel_sigma: float = 0.0
    display_ranges: dict[str, DisplayRangeConfig] = field(default_factory=dict)


@dataclass
class MapsConfig:
    fluxmap: Optional[str] = None
    velmap: Optional[str] = None
    sigmap: Optional[str] = None


@dataclass
class DiscIndependentConfig:
    v_grid_kms: List[float] = field(default_factory=lambda: [0.0, 500.0, 10.0])

@dataclass
class DiscKeplerConfig:
    mbh_grid_msun: List[float] = field(default_factory=lambda: [1.0e6, 1.0e11])
    n_geom: int = 50

@dataclass
class DiscNSCConfig:
    re_pc: float = 5.0
    a_grid: List[float] = field(default_factory=lambda: [1.0e-3, 1.0e3])
    n_geom: int = 50

@dataclass
class DiscPlummerConfig:
    a_pc: float = 4.0
    m0_grid_msun: List[float] = field(default_factory=lambda: [1.0e6, 1.0e11])
    n_geom: int = 50

@dataclass
class DiscArctanConfig:
    rt_arcsec: Optional[float] = None
    vmax_grid_kms: List[float] = field(default_factory=lambda: [0.0, 500.0, 10.0])

@dataclass
class DiscFitConfig:
    mode: DiscModelMode = "independent"
    radius_range_arcsec: List[float] = field(default_factory=lambda: [0.0, 40.0])
    num_shells: int = 30
    pa_deg: Optional[float] = None
    pa_unc_deg: Optional[float] = None
    beta_grid_deg: List[float] = field(default_factory=lambda: [50, 100, 5])

    independent: DiscIndependentConfig = field(default_factory=DiscIndependentConfig)
    kepler: DiscKeplerConfig = field(default_factory=DiscKeplerConfig)
    nsc: DiscNSCConfig = field(default_factory=DiscNSCConfig)
    plummer: DiscPlummerConfig = field(default_factory=DiscPlummerConfig)
    arctan: DiscArctanConfig = field(default_factory=DiscArctanConfig)


@dataclass
class OutflowFitConfig:
    radius_range_arcsec: List[float] = field(default_factory=lambda: [0.0, 30.0])
    num_shells: int = 30
    pa_deg: float = 105.0
    opening_deg: float = 120.0
    double_cone: bool = True
    mask_mode: MaskMode = "bicone"
    beta_grid_deg: List[float] = field(default_factory=lambda: [50, 130, 5])
    v_grid_kms: List[float] = field(default_factory=lambda: [100, 1300, 20])

@dataclass
class FitConfig:
    component_mode: FitMode = "disk_then_outflow"
    disc: DiscFitConfig = field(default_factory=DiscFitConfig)
    outflow: OutflowFitConfig = field(default_factory=OutflowFitConfig)




@dataclass
class AdvancedConfig:
    check_masking_before_fitting: bool = False
    use_crps: bool = False
    crps_qgrid: List[float] = field(default_factory=lambda: [0.01, 0.99])
    perc_disc: List[float] = field(default_factory=lambda: [0.05, 0.95])
    perc_out: List[float] = field(default_factory=lambda: [0.01, 0.99])
    perc_weights: List[float] = field(default_factory=lambda: [1, 1])
    npt: int = 200_000

    use_global_beta_disc: bool = True
    disc_theta_range: List[List[float]] = field(default_factory=lambda: [[0, 1]])
    disc_phi_range: List[List[float]] = field(default_factory=lambda: [[0, 360]])
    disc_zeta_range_mode: str = "auto_from_psf"

    use_global_beta_out: bool = True

    mask_disk_with_outflow: bool = True
    mask_mode: str = "zero"
    do_final_combined_model_plot: bool = True
    outflow_axis_sign: int = 1
    resid_ranges: List[float] = field(default_factory=lambda: [0.15, 55, 55])
    compute_escape_fraction: bool = False
    save_escape_fraction_table: bool = True


@dataclass
class OutputConfig:
    save_plots: bool = True
    show_plots: bool = False
    save_summary_json: bool = True
    save_run_config_copy: bool = True
    overwrite: bool = True


@dataclass
class AppConfig:
    paths: PathsConfig
    input: InputConfig
    target: TargetConfig
    line: LineConfig
    processing: ProcessingConfig
    maps: MapsConfig
    fit: FitConfig
    advanced: AdvancedConfig
    output: OutputConfig


def _as_path(value: Any) -> Path:
    return Path(value).expanduser().resolve()

def _as_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    return [float(x) for x in list(value)]


def _as_int(value: Any) -> int:
    return int(value)

def _as_float_or_none(value: Any) -> Optional[float]:
    return None if value is None else float(value)



def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    fit_raw = raw["fit"]
    disc_raw = fit_raw.get("disc", {})
    out_raw = fit_raw.get("outflow", {})

    # ----------------------------
    # Processing block
    # ----------------------------
    proc_raw = raw.get("processing", {})

    display_ranges_raw = proc_raw.get("display_ranges", {})
    display_ranges = {
        k: DisplayRangeConfig(**v)
        for k, v in display_ranges_raw.items()
    }

    for key in ["flux", "vel", "sig"]:
        display_ranges.setdefault(key, DisplayRangeConfig())

    processing = ProcessingConfig(
        sn_thresh=float(proc_raw.get("sn_thresh", 3.0)) if proc_raw.get("sn_thresh", 3.0) is not None else None,
        nrebin=int(proc_raw.get("nrebin", 1)),
        xrange=proc_raw.get("xrange"),
        yrange=proc_raw.get("yrange"),
        pixel_scale_arcsec_manual=_as_float_or_none(proc_raw.get("pixel_scale_arcsec_manual")),
        psf_sigma=float(proc_raw.get("psf_sigma", 1.0)),
        lsf_sigma=float(proc_raw.get("lsf_sigma", 72.0)),
        vel_sigma=float(proc_raw.get("vel_sigma", 0.0)),
        display_ranges=display_ranges,
    )

    # ----------------------------
    # Full app config
    # ----------------------------
    cfg = AppConfig(
        paths=PathsConfig(
            data_dir=_as_path(raw["paths"]["data_dir"]),
            ancillary_dir=_as_path(raw["paths"]["ancillary_dir"]),
            output_dir=_as_path(raw["paths"]["output_dir"]),
        ),
        input=InputConfig(**raw["input"]),
        target=TargetConfig(**raw["target"]),
        line=LineConfig(**raw["line"]),
        processing=processing,
        maps=MapsConfig(**raw.get("maps", {})),
        fit=FitConfig(
            component_mode=fit_raw.get("component_mode", "disk_then_outflow"),
            disc=DiscFitConfig(
                mode=disc_raw.get("mode", "independent"),
                radius_range_arcsec=_as_float_list(disc_raw.get("radius_range_arcsec", [0.0, 40.0])),
                num_shells=_as_int(disc_raw.get("num_shells", 30)),
                pa_deg=_as_float_or_none(disc_raw.get("pa_deg", None)),
                pa_unc_deg=_as_float_or_none(disc_raw.get("pa_unc_deg", None)),
                beta_grid_deg=_as_float_list(disc_raw.get("beta_grid_deg", [50, 100, 5])),

                independent=DiscIndependentConfig(
                    v_grid_kms=_as_float_list(
                        disc_raw.get("independent", {}).get("v_grid_kms", [0.0, 500.0, 10.0])
                    )
                ),

                kepler=DiscKeplerConfig(
                    mbh_grid_msun=_as_float_list(
                        disc_raw.get("kepler", {}).get("mbh_grid_msun", [1.0e6, 1.0e11])
                    ),
                    n_geom=_as_int(disc_raw.get("kepler", {}).get("n_geom", 50)),
                ),

                nsc=DiscNSCConfig(
                    re_pc=float(disc_raw.get("nsc", {}).get("re_pc", 5.0)),
                    a_grid=_as_float_list(
                        disc_raw.get("nsc", {}).get("a_grid", [1.0e-3, 1.0e3])
                    ),
                    n_geom=_as_int(disc_raw.get("nsc", {}).get("n_geom", 50)),
                ),

                plummer=DiscPlummerConfig(
                    a_pc=float(disc_raw.get("plummer", {}).get("a_pc", 4.0)),
                    m0_grid_msun=_as_float_list(
                        disc_raw.get("plummer", {}).get("m0_grid_msun", [1.0e6, 1.0e11])
                    ),
                    n_geom=_as_int(disc_raw.get("plummer", {}).get("n_geom", 50)),
                ),

                arctan=DiscArctanConfig(
                    rt_arcsec=_as_float_or_none(disc_raw.get("arctan", {}).get("rt_arcsec", None)),
                    vmax_grid_kms=_as_float_list(
                        disc_raw.get("arctan", {}).get("vmax_grid_kms", [0.0, 500.0, 10.0])
                    ),
                ),
            ),

            outflow=OutflowFitConfig(
                radius_range_arcsec=_as_float_list(out_raw.get("radius_range_arcsec", [0.0, 30.0])),
                num_shells=_as_int(out_raw.get("num_shells", 30)),
                pa_deg=float(out_raw.get("pa_deg", 105.0)),
                opening_deg=float(out_raw.get("opening_deg", 120.0)),
                double_cone=bool(out_raw.get("double_cone", True)),
                mask_mode=str(out_raw.get("mask_mode", "bicone")),
                beta_grid_deg=_as_float_list(out_raw.get("beta_grid_deg", [50, 130, 5])),
                v_grid_kms=_as_float_list(out_raw.get("v_grid_kms", [100.0, 1300.0, 20.0])),
            ),
        ),
        advanced=AdvancedConfig(**raw.get("advanced", {})),
        output=OutputConfig(**raw.get("output", {})),
    )

    validate_config(cfg)
    return cfg




def validate_config(cfg: AppConfig) -> None:
    # --------------------------------
    # Top-level fit mode
    # --------------------------------
    if cfg.fit.component_mode not in {"disk", "outflow", "disk_then_outflow"}:
        raise ValueError("fit.component_mode must be 'disk', 'outflow', or 'disk_then_outflow'.")

    # --------------------------------
    # General processing checks
    # --------------------------------
    if cfg.processing.nrebin < 1:
        raise ValueError("processing.nrebin must be >= 1.")

    cube_path = cfg.paths.data_dir / cfg.input.cube_file
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube file not found: {cube_path}")

    if cfg.input.sn_map is not None:
        sn_path = cfg.paths.ancillary_dir / cfg.input.sn_map
        if not sn_path.exists():
            raise FileNotFoundError(f"SN map not found: {sn_path}")

    # --------------------------------
    # Disc config checks
    # --------------------------------

    
    disc = cfg.fit.disc
    out = cfg.fit.outflow

    allowed_disc_modes = {"independent", "disk_kepler", "NSC", "Plummer", "disk_arctan"}
    if disc.mode not in allowed_disc_modes:
        raise ValueError(f"fit.disc.mode must be one of {sorted(allowed_disc_modes)}.")

    if len(disc.radius_range_arcsec) != 2:
        raise ValueError("fit.disc.radius_range_arcsec must have [rmin, rmax].")

    if disc.radius_range_arcsec[1] <= disc.radius_range_arcsec[0]:
        raise ValueError("fit.disc.radius_range_arcsec must satisfy rmax > rmin.")

    if disc.num_shells < 1:
        raise ValueError("fit.disc.num_shells must be >= 1.")

    if len(disc.beta_grid_deg) != 3:
        raise ValueError("fit.disc.beta_grid_deg must have [min, max, step].")

    if disc.beta_grid_deg[2] <= 0:
        raise ValueError("fit.disc.beta_grid_deg step must be > 0.")

    if out.beta_grid_deg[2] <= 0:
        raise ValueError("fit.outflow.beta_grid_deg step must be > 0.")

    if disc.mode == "independent":
        if len(disc.independent.v_grid_kms) != 3:
            raise ValueError("fit.disc.independent.v_grid_kms must have [min, max, step].")

        if disc.independent.v_grid_kms[2] <= 0:
            raise ValueError("fit.disc.independent.v_grid_kms step must be > 0.")

    elif disc.mode == "disk_kepler":
        if len(disc.kepler.mbh_grid_msun) != 2:
            raise ValueError("fit.disc.kepler.mbh_grid_msun must have [min, max].")
        if disc.kepler.n_geom < 2:
            raise ValueError("fit.disc.kepler.n_geom must be >= 2.")
        if disc.kepler.mbh_grid_msun[0] <= 0 or disc.kepler.mbh_grid_msun[1] <= 0:
            raise ValueError("fit.disc.kepler.mbh_grid_msun values must be > 0 for geomspace.")

    elif disc.mode == "NSC":
        if disc.nsc.re_pc is None or disc.nsc.re_pc <= 0:
            raise ValueError("fit.disc.nsc.re_pc must be > 0.")
        if len(disc.nsc.a_grid) != 2:
            raise ValueError("fit.disc.nsc.a_grid must have [min, max].")
        if disc.nsc.n_geom < 2:
            raise ValueError("fit.disc.nsc.n_geom must be >= 2.")
        if disc.nsc.a_grid[0] <= 0 or disc.nsc.a_grid[1] <= 0:
            raise ValueError("fit.disc.nsc.a_grid values must be > 0 for geomspace.")

    elif disc.mode == "Plummer":
        if disc.plummer.a_pc is None or disc.plummer.a_pc <= 0:
            raise ValueError("fit.disc.plummer.a_pc must be > 0.")
        if len(disc.plummer.m0_grid_msun) != 2:
            raise ValueError("fit.disc.plummer.m0_grid_msun must have [min, max].")
        if disc.plummer.n_geom < 2:
            raise ValueError("fit.disc.plummer.n_geom must be >= 2.")
        if disc.plummer.m0_grid_msun[0] <= 0 or disc.plummer.m0_grid_msun[1] <= 0:
            raise ValueError("fit.disc.plummer.m0_grid_msun values must be > 0 for geomspace.")

    elif disc.mode == "disk_arctan":
        if disc.arctan.rt_arcsec is None or disc.arctan.rt_arcsec <= 0:
            raise ValueError("fit.disc.arctan.rt_arcsec must be > 0.")
        if len(disc.arctan.vmax_grid_kms) != 3:
            raise ValueError("fit.disc.arctan.vmax_grid_kms must have [min, max, step].")

        if disc.arctan.vmax_grid_kms[2] <= 0:
            raise ValueError("fit.disc.arctan.vmax_grid_kms step must be > 0.")

    # --------------------------------
    # Outflow config checks
    # --------------------------------
    if len(out.radius_range_arcsec) != 2:
        raise ValueError("fit.outflow.radius_range_arcsec must have [rmin, rmax].")

    if out.radius_range_arcsec[1] <= out.radius_range_arcsec[0]:
        raise ValueError("fit.outflow.radius_range_arcsec must satisfy rmax > rmin.")

    if out.num_shells < 1:
        raise ValueError("fit.outflow.num_shells must be >= 1.")

    if out.mask_mode not in {"single", "bicone"}:
        raise ValueError("fit.outflow.mask_mode must be 'single' or 'bicone'.")

    if len(out.beta_grid_deg) != 3:
        raise ValueError("fit.outflow.beta_grid_deg must have [min, max, step].")

    if len(out.v_grid_kms) != 3:
        raise ValueError("fit.outflow.v_grid_kms must have [min, max, step].")
    if out.v_grid_kms[2] <= 0:
        raise ValueError("fit.outflow.v_grid_kms step must be > 0.")
    if out.opening_deg <= 0 or out.opening_deg > 180:
        raise ValueError("fit.outflow.opening_deg must be in the range (0, 180].")






