from __future__ import annotations

import numpy as np
from astropy.io import fits
import pytest

from moka3d import moka3d_source as km
from moka3d.config import validate_config
from moka3d.pipeline import run_pipeline

from ._synthetic import DEFAULT_BUNIT, SyntheticCubeCase
from .test_smoke import _build_smoke_config, _disable_plot_writes


class _StopOnEnergeticsBuild(RuntimeError):
    """Sentinel used to stop the pipeline at the first energetics-builder call."""


def test_flux_unit_scale_parses_supported_bunit() -> None:
    scale = km.flux_unit_scale_from_bunit(DEFAULT_BUNIT)
    assert scale is not None
    assert np.isfinite(scale)
    assert scale > 0.0
    assert scale == pytest.approx(1.0e-20, rel=0.0, abs=1.0e-30)


def test_flux_unit_scale_returns_none_for_unsupported_bunit() -> None:
    assert km.flux_unit_scale_from_bunit("definitely_not_a_supported_flux_density_unit") is None


def test_pipeline_passes_supported_flux_unit_scale_into_energetics_builder(
    fx_outflow_energetics_wl_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_outflow_energetics_wl_small.data_dir.parent / "outputs_energetics_units_supported"
    cfg = _build_smoke_config(
        fx_outflow_energetics_wl_small,
        component_mode="outflow",
        output_root=output_root,
        compute_energetics=True,
    )
    validate_config(cfg)

    captured: dict[str, float] = {}

    def build_wrapper(*args, **kwargs):
        captured["flux_unit_scale"] = float(kwargs["flux_unit_scale"])
        raise _StopOnEnergeticsBuild

    monkeypatch.setattr(km, "build_outflow_energetics_profile", build_wrapper)

    with pytest.raises(_StopOnEnergeticsBuild):
        run_pipeline(cfg)

    assert captured["flux_unit_scale"] == pytest.approx(1.0e-20, rel=0.0, abs=1.0e-30)


def test_pipeline_does_not_build_physical_energetics_with_unparseable_bunit(
    fx_outflow_energetics_wl_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)

    with fits.open(fx_outflow_energetics_wl_small.cube_path, mode="update") as hdul:
        hdul[0].header["BUNIT"] = "definitely_not_a_supported_flux_density_unit"
        hdul.flush()

    output_root = fx_outflow_energetics_wl_small.data_dir.parent / "outputs_energetics_units_bad_bunit"
    cfg = _build_smoke_config(
        fx_outflow_energetics_wl_small,
        component_mode="outflow",
        output_root=output_root,
        compute_energetics=True,
    )
    validate_config(cfg)

    assert km.flux_unit_scale_from_bunit("definitely_not_a_supported_flux_density_unit") is None

    def fail_if_called(*args, **kwargs):
        raise _StopOnEnergeticsBuild

    monkeypatch.setattr(km, "build_outflow_energetics_profile", fail_if_called)

    run_pipeline(cfg)
