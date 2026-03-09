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
    agn_ra: Optional[float] = None
    agn_dec: Optional[float] = None
    center_mode: CenterMode = None
    center_xy_manual: Optional[List[int]] = None
    redshift: float = 0.0


@dataclass
class LineConfig:
    wavelength_line: float
    wavelength_line_unit: str = "Angstrom"


@dataclass
class ProcessingConfig:
    sn_thresh: float = 3.0
    nrebin: int = 1
    xrange: Optional[List[float]] = None
    yrange: Optional[List[float]] = None
    percentile_shown_mom_maps: List[List[float]] = field(default_factory=lambda: [[1, 99], [5, 95], [1, 99]])
    psf_sigma: float = 1.0
    lsf_sigma: float = 72.0
    vel_sigma: float = 0.0
    logradius: bool = False


@dataclass
class MapsConfig:
    fluxmap: Optional[str] = None
    velmap: Optional[str] = None
    sigmap: Optional[str] = None


@dataclass
class FitConfig:
    component_mode: FitMode = "disk_then_outflow"
    radius_range_model_disc: List[float] = field(default_factory=lambda: [0.0, 40.0])
    radius_range_model_out: List[float] = field(default_factory=lambda: [0.0, 30.0])
    num_shells_disc: int = 30
    num_shells_out: int = 30
    beta_grid_disc: List[float] = field(default_factory=lambda: [50, 100, 5])
    v_grid_disc: List[float] = field(default_factory=lambda: [0, 400, 20])
    outflow_pa_deg: float = 105.0
    outflow_opening_deg: float = 120.0
    outflow_double_cone: bool = True
    outflow_mask_mode: MaskMode = "bicone"
    beta_grid_out: List[float] = field(default_factory=lambda: [50, 130, 5])
    v_grid_out: List[float] = field(default_factory=lambda: [100, 1300, 20])
    disc_pa_deg: float | None = None


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
    disc_fit_mode: str = "independent"
    disc_geometry: str = "cylindrical"
    disc_theta_range: List[List[float]] = field(default_factory=lambda: [[0, 1]])
    disc_phi_range: List[List[float]] = field(default_factory=lambda: [[0, 360]])
    disc_zeta_range_mode: str = "auto_from_psf"
    disc_double_cone: bool = False

    use_global_beta_out: bool = True
    outflow_fit_mode: str = "independent"
    out_geometry: str = "spherical"

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


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig(
        paths=PathsConfig(
            data_dir=_as_path(raw["paths"]["data_dir"]),
            ancillary_dir=_as_path(raw["paths"]["ancillary_dir"]),
            output_dir=_as_path(raw["paths"]["output_dir"]),
        ),
        input=InputConfig(**raw["input"]),
        target=TargetConfig(**raw["target"]),
        line=LineConfig(**raw["line"]),
        processing=ProcessingConfig(**raw.get("processing", {})),
        maps=MapsConfig(**raw.get("maps", {})),
        fit=FitConfig(**raw["fit"]),
        advanced=AdvancedConfig(**raw.get("advanced", {})),
        output=OutputConfig(**raw.get("output", {})),
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    if cfg.fit.component_mode not in {"disk", "outflow", "disk_then_outflow"}:
        raise ValueError("fit.component_mode must be 'disk', 'outflow', or 'disk_then_outflow'.")

    if cfg.fit.outflow_mask_mode not in {"single", "bicone"}:
        raise ValueError("fit.outflow_mask_mode must be 'single' or 'bicone'.")

    if len(cfg.fit.beta_grid_disc) != 3:
        raise ValueError("fit.beta_grid_disc must have [min, max, step].")
    if len(cfg.fit.v_grid_disc) != 3:
        raise ValueError("fit.v_grid_disc must have [min, max, step].")
    if len(cfg.fit.beta_grid_out) != 3:
        raise ValueError("fit.beta_grid_out must have [min, max, step].")
    if len(cfg.fit.v_grid_out) != 3:
        raise ValueError("fit.v_grid_out must have [min, max, step].")

    if cfg.processing.nrebin < 1:
        raise ValueError("processing.nrebin must be >= 1.")

    cube_path = cfg.paths.data_dir / cfg.input.cube_file
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube file not found: {cube_path}")

    if cfg.input.sn_map is not None:
        sn_path = cfg.paths.ancillary_dir / cfg.input.sn_map
        if not sn_path.exists():
            raise FileNotFoundError(f"SN map not found: {sn_path}")