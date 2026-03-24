from __future__ import annotations

import json
from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt
import pytest

from moka3d.config import (
    AdvancedConfig,
    AppConfig,
    DisplayRangeConfig,
    DiscFitConfig,
    DiscIndependentConfig,
    FitConfig,
    InputConfig,
    LineConfig,
    MapsConfig,
    OutflowFitConfig,
    OutputConfig,
    PathsConfig,
    ProcessingConfig,
    TargetConfig,
    validate_config,
)
from moka3d.pipeline import run_pipeline

from ._synthetic import SyntheticCubeCase


def _build_smoke_config(
    case: SyntheticCubeCase,
    *,
    component_mode: str,
    output_root: Path,
    compute_energetics: bool = False,
    save_plots: bool = False,
) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            data_dir=case.data_dir,
            ancillary_dir=case.ancillary_dir,
            output_dir=output_root,
        ),
        input=InputConfig(
            cube_file=case.cube_file,
            sn_map=case.sn_map_file,
            save_all_outputs=True,
            ne_map=case.ne_map_file,
            ne_outflow=None,
        ),
        target=TargetConfig(
            agn_ra=None,
            agn_dec=None,
            center_mode=None,
            center_xy_manual=[int(case.center_xy[0]), int(case.center_xy[1])],
            redshift=float(case.redshift),
        ),
        line=LineConfig(
            wavelength_line=float(case.line_value),
            wavelength_line_unit=str(case.line_unit),
        ),
        processing=ProcessingConfig(
            sn_thresh=3.0,
            nrebin=1,
            xrange=None,
            yrange=None,
            pixel_scale_arcsec_manual=None,
            psf_sigma=1.0,
            lsf_sigma=30.0,
            vel_sigma=0.0,
            display_ranges={
                "flux": DisplayRangeConfig(),
                "vel": DisplayRangeConfig(),
                "sig": DisplayRangeConfig(),
            },
        ),
        maps=MapsConfig(),
        fit=FitConfig(
            component_mode=component_mode,
            disc=DiscFitConfig(
                mode="independent",
                radius_range_arcsec=[0.0, 1.0],
                num_shells=1,
                pa_deg=float(case.expected.get("disk_pa_deg", 35.0)),
                pa_unc_deg=None,
                beta_grid_deg=[60.0, 60.0, 1.0],
                independent=DiscIndependentConfig(v_grid_kms=[140.0, 140.0, 1.0]),
            ),
            outflow=OutflowFitConfig(
                radius_range_arcsec=[0.0, 1.0],
                num_shells=2 if compute_energetics else 1,
                pa_deg=float(case.expected.get("outflow_pa_deg", 90.0)),
                opening_deg=float(case.expected.get("outflow_opening_deg", 100.0)),
                double_cone=True,
                mask_mode="bicone",
                beta_grid_deg=[60.0, 60.0, 1.0],
                v_grid_kms=[220.0, 220.0, 1.0],
            ),
        ),
        advanced=AdvancedConfig(
            check_masking_before_fitting=False,
            use_crps=False,
            npt=500,
            compute_escape_fraction=False,
            save_escape_fraction_table=False,
            compute_energetics=compute_energetics,
            save_energetics_table=True,
        ),
        output=OutputConfig(
            save_plots=save_plots,
            show_plots=False,
            save_summary_json=True,
            save_run_config_copy=False,
            overwrite=True,
        ),
    )


def _disable_plot_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _close_only(output_path=None, show=False, close=True) -> None:
        if close:
            plt.close("all")

    monkeypatch.setattr("moka3d.pipeline.finalize_figure", _close_only)


def _single_run_dir(output_root: Path) -> Path:
    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    return run_dirs[0]


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    assert summary_path.exists()
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _assert_common_artifacts(
    run_dir: Path,
    *,
    expected_cube_shape: tuple[int, int, int],
    summary: dict,
) -> None:
    assert (run_dir / "moka3d.log").exists()
    assert summary["input_cube"].endswith(".fits")
    assert summary["center_method"] == "manual"
    assert "fit_component_mode" in summary

    cube_files = sorted(run_dir.glob("bestfit*_weighted_cube.fits"))
    assert len(cube_files) == 1

    with fits.open(cube_files[0]) as hdul:
        assert hdul[0].data is not None
        assert hdul[0].data.shape == expected_cube_shape

    maps_path = run_dir / "bestfit_moment_maps.fits"
    assert maps_path.exists()
    with fits.open(maps_path) as hdul:
        assert len(hdul) == 10
        for hdu in hdul[1:]:
            assert hdu.data is not None
            assert hdu.data.shape == expected_cube_shape[1:]


def test_disk_wavelength_smoke(
    fx_disk_wl_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_disk_wl_small.data_dir.parent / "outputs_disk"
    cfg = _build_smoke_config(
        fx_disk_wl_small,
        component_mode="disk",
        output_root=output_root,
    )

    validate_config(cfg)
    summary = run_pipeline(cfg)
    run_dir = _single_run_dir(output_root)
    summary_disk = _load_summary(run_dir)

    assert summary["fit_component_mode"] == "disk"
    assert summary_disk["fit_component_mode"] == "disk"
    _assert_common_artifacts(run_dir, expected_cube_shape=fx_disk_wl_small.shape, summary=summary_disk)


def test_outflow_wavelength_smoke(
    fx_outflow_wl_bicone_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_outflow_wl_bicone_small.data_dir.parent / "outputs_outflow"
    cfg = _build_smoke_config(
        fx_outflow_wl_bicone_small,
        component_mode="outflow",
        output_root=output_root,
    )

    validate_config(cfg)
    summary = run_pipeline(cfg)
    run_dir = _single_run_dir(output_root)
    summary_out = _load_summary(run_dir)

    assert summary["fit_component_mode"] == "outflow"
    assert summary_out["outflow_topology"] == "bicone"
    _assert_common_artifacts(run_dir, expected_cube_shape=fx_outflow_wl_bicone_small.shape, summary=summary_out)


def test_combined_disk_outflow_smoke(
    fx_combo_wl_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_combo_wl_small.data_dir.parent / "outputs_combo"
    cfg = _build_smoke_config(
        fx_combo_wl_small,
        component_mode="disk_then_outflow",
        output_root=output_root,
    )

    validate_config(cfg)
    summary = run_pipeline(cfg)
    run_dir = _single_run_dir(output_root)
    summary_combo = _load_summary(run_dir)

    assert summary["fit_component_mode"] == "disk_then_outflow"
    assert summary_combo["fit_component_mode"] == "disk_then_outflow"
    _assert_common_artifacts(run_dir, expected_cube_shape=fx_combo_wl_small.shape, summary=summary_combo)


def test_frequency_cube_smoke(
    fx_disk_freq_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_disk_freq_small.data_dir.parent / "outputs_freq"
    cfg = _build_smoke_config(
        fx_disk_freq_small,
        component_mode="disk",
        output_root=output_root,
    )

    validate_config(cfg)
    summary = run_pipeline(cfg)
    run_dir = _single_run_dir(output_root)
    summary_freq = _load_summary(run_dir)

    assert summary["fit_component_mode"] == "disk"
    assert summary_freq["spec_unit"] == "Hz"
    _assert_common_artifacts(run_dir, expected_cube_shape=fx_disk_freq_small.shape, summary=summary_freq)


def test_outflow_energetics_smoke(
    fx_outflow_energetics_wl_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_outflow_energetics_wl_small.data_dir.parent / "outputs_energetics"
    cfg = _build_smoke_config(
        fx_outflow_energetics_wl_small,
        component_mode="outflow",
        output_root=output_root,
        compute_energetics=True,
    )

    validate_config(cfg)
    summary = run_pipeline(cfg)
    run_dir = _single_run_dir(output_root)
    summary_ener = _load_summary(run_dir)

    assert summary["compute_energetics"] is True
    assert summary_ener["compute_energetics"] is True
    energetics_tables = sorted(run_dir.glob("outflow_energetics_*.fits"))
    assert energetics_tables
    with fits.open(energetics_tables[0]) as hdul:
        table_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        assert table_hdu.data is not None
        assert len(table_hdu.data) >= 1
    _assert_common_artifacts(run_dir, expected_cube_shape=fx_outflow_energetics_wl_small.shape, summary=summary_ener)


def test_reporting_outputs_smoke(fx_reporting_reuse: SyntheticCubeCase) -> None:
    output_root = fx_reporting_reuse.data_dir.parent / "outputs_reporting"
    cfg = _build_smoke_config(
        fx_reporting_reuse,
        component_mode="disk",
        output_root=output_root,
        save_plots=True,
    )

    validate_config(cfg)
    summary = run_pipeline(cfg)
    run_dir = _single_run_dir(output_root)
    summary_reporting = _load_summary(run_dir)

    assert summary["fit_component_mode"] == "disk"
    png_files = sorted(run_dir.glob("*.png"))
    assert png_files
    _assert_common_artifacts(run_dir, expected_cube_shape=fx_reporting_reuse.shape, summary=summary_reporting)
