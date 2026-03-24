from __future__ import annotations

import numpy as np
import astropy.units as u
import pytest

from moka3d import moka3d_source as km
from moka3d.config import validate_config
from moka3d.pipeline import run_pipeline

from ._synthetic import DEFAULT_FREQ_LINE_HZ, DEFAULT_REDSHIFT, SyntheticCubeCase, frequency_axis_hz
from .test_smoke import _build_smoke_config, _disable_plot_writes


def test_velocity_axis_frequency_defaults_to_radio_convention() -> None:
    v_true_kms = np.array([-30000.0, 0.0, 30000.0], dtype=float)
    spec_coord = frequency_axis_hz(
        v_true_kms,
        rest_frequency_hz=DEFAULT_FREQ_LINE_HZ,
        redshift=DEFAULT_REDSHIFT,
    ) * u.Hz

    vel_default, spec_kind_default, _ = km.velocity_axis_from_spectral_coord(
        spec_coord,
        u.Hz,
        line_value=DEFAULT_FREQ_LINE_HZ,
        line_unit="Hz",
        redshift=DEFAULT_REDSHIFT,
    )
    vel_radio, spec_kind_radio, _ = km.velocity_axis_from_spectral_coord(
        spec_coord,
        u.Hz,
        line_value=DEFAULT_FREQ_LINE_HZ,
        line_unit="Hz",
        redshift=DEFAULT_REDSHIFT,
        convention="radio",
    )
    vel_optical, _, _ = km.velocity_axis_from_spectral_coord(
        spec_coord,
        u.Hz,
        line_value=DEFAULT_FREQ_LINE_HZ,
        line_unit="Hz",
        redshift=DEFAULT_REDSHIFT,
        convention="optical",
    )

    assert spec_kind_default == "frequency"
    assert spec_kind_radio == "frequency"
    np.testing.assert_allclose(vel_default, vel_radio, rtol=0.0, atol=1e-9)
    assert np.max(np.abs(vel_radio - vel_optical)) > 1000.0


def test_velocity_axis_frequency_explicit_optical_override_is_respected() -> None:
    v_true_kms = np.array([-30000.0, 0.0, 30000.0], dtype=float)
    spec_coord = frequency_axis_hz(
        v_true_kms,
        rest_frequency_hz=DEFAULT_FREQ_LINE_HZ,
        redshift=DEFAULT_REDSHIFT,
    ) * u.Hz

    vel_radio, _, _ = km.velocity_axis_from_spectral_coord(
        spec_coord,
        u.Hz,
        line_value=DEFAULT_FREQ_LINE_HZ,
        line_unit="Hz",
        redshift=DEFAULT_REDSHIFT,
        convention="radio",
    )
    vel_optical, spec_kind, _ = km.velocity_axis_from_spectral_coord(
        spec_coord,
        u.Hz,
        line_value=DEFAULT_FREQ_LINE_HZ,
        line_unit="Hz",
        redshift=DEFAULT_REDSHIFT,
        convention="optical",
    )

    assert spec_kind == "frequency"
    assert not np.allclose(vel_radio, vel_optical, rtol=0.0, atol=1e-9)


class _StopAfterVelocityAxis(RuntimeError):
    """Sentinel used to stop the pipeline immediately after convention selection is observed."""


def test_pipeline_defaults_frequency_cubes_to_radio_convention(
    fx_disk_freq_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_disk_freq_small.data_dir.parent / "outputs_freq_doppler"
    cfg = _build_smoke_config(
        fx_disk_freq_small,
        component_mode="disk",
        output_root=output_root,
    )
    validate_config(cfg)

    captured: dict[str, object] = {}
    original_velocity_axis = km.velocity_axis_from_spectral_coord

    def velocity_axis_wrapper(spec_coord, spec_unit, **kwargs):
        captured["spec_unit"] = spec_unit
        captured["convention"] = kwargs.get("convention")
        if spec_unit.is_equivalent(u.Hz):
            raise _StopAfterVelocityAxis
        return original_velocity_axis(spec_coord, spec_unit, **kwargs)

    monkeypatch.setattr(km, "velocity_axis_from_spectral_coord", velocity_axis_wrapper)

    with pytest.raises(_StopAfterVelocityAxis):
        run_pipeline(cfg)

    assert captured["spec_unit"].is_equivalent(u.Hz)
    assert captured["convention"] == "radio"


def test_pipeline_respects_explicit_frequency_convention_override(
    fx_disk_freq_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_disk_freq_small.data_dir.parent / "outputs_freq_doppler_override"
    cfg = _build_smoke_config(
        fx_disk_freq_small,
        component_mode="disk",
        output_root=output_root,
    )
    cfg.line.doppler_convention = "optical"
    validate_config(cfg)

    captured: dict[str, object] = {}
    original_velocity_axis = km.velocity_axis_from_spectral_coord

    def velocity_axis_wrapper(spec_coord, spec_unit, **kwargs):
        captured["spec_unit"] = spec_unit
        captured["convention"] = kwargs.get("convention")
        if spec_unit.is_equivalent(u.Hz):
            raise _StopAfterVelocityAxis
        return original_velocity_axis(spec_coord, spec_unit, **kwargs)

    monkeypatch.setattr(km, "velocity_axis_from_spectral_coord", velocity_axis_wrapper)

    with pytest.raises(_StopAfterVelocityAxis):
        run_pipeline(cfg)

    assert captured["spec_unit"].is_equivalent(u.Hz)
    assert captured["convention"] == "optical"
