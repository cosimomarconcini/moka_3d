from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from moka3d.config import validate_config
from moka3d.pipeline import run_pipeline
from moka3d import moka3d_source as km

from ._synthetic import SyntheticCubeCase
from .test_smoke import _build_smoke_config, _disable_plot_writes


class _StopOnFirstOutflowEval(RuntimeError):
    """Sentinel used to stop the pipeline once the combined-fit handoff is observed."""


class _StopOnSecondLoopEval(RuntimeError):
    """Sentinel used to stop a direct fit-loop regression after two evaluations."""


def test_combined_outflow_fit_retains_disc_cube(
    fx_combo_wl_small: SyntheticCubeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_plot_writes(monkeypatch)
    output_root = fx_combo_wl_small.data_dir.parent / "outputs_regression_combo"
    cfg = _build_smoke_config(
        fx_combo_wl_small,
        component_mode="disk_then_outflow",
        output_root=output_root,
    )

    validate_config(cfg)

    original_fit_gridsearch = km.fit_gridsearch_component
    original_eval_kappa = km.eval_kappa_for_model
    stage: dict[str, str | None] = {"label": None}
    captured: dict[str, object] = {}

    def fit_gridsearch_wrapper(**kwargs):
        label = str(kwargs.get("verbose_label", ""))
        if label.startswith("OUTFLOW"):
            stage["label"] = label
            try:
                return original_fit_gridsearch(**kwargs)
            finally:
                stage["label"] = None
        return original_fit_gridsearch(**kwargs)

    def eval_kappa_wrapper(combined_model_cube, beta, gamma_model_deg, *, ctx=None):
        if stage["label"] is not None:
            disc_cube = km._FIT_CTX.get("disc_cube", None)
            captured["stage"] = stage["label"]
            captured["disc_cube"] = None if disc_cube is None else np.array(disc_cube, copy=True)
            captured["combined_cube_shape"] = np.asarray(combined_model_cube).shape
            raise _StopOnFirstOutflowEval
        return original_eval_kappa(combined_model_cube, beta, gamma_model_deg, ctx=ctx)

    monkeypatch.setattr(km, "fit_gridsearch_component", fit_gridsearch_wrapper)
    monkeypatch.setattr(km, "eval_kappa_for_model", eval_kappa_wrapper)

    with pytest.raises(_StopOnFirstOutflowEval):
        run_pipeline(cfg)

    assert captured["stage"] is not None
    assert str(captured["stage"]).startswith("OUTFLOW")

    disc_cube = captured["disc_cube"]
    assert disc_cube is not None
    assert disc_cube.shape == fx_combo_wl_small.shape
    assert np.isfinite(disc_cube).any()
    assert np.nansum(disc_cube) > 0.0


def test_fit_gridsearch_component_ignores_fit_ctx_disc_cube_poisoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (5, 4, 4)
    disc_cube_input = np.full(shape, 3.0, dtype=float)
    poisoned_disc_cube = np.full(shape, 99.0, dtype=float)
    obs_stub = SimpleNamespace(cube={"data": np.zeros(shape, dtype=float)})
    captured: list[np.ndarray] = []

    def build_v_grid_stub(geometry: str, fit_mode: str, **kwargs):
        return np.array([10.0, 20.0], dtype=float), None, "v"

    def build_model_stub(beta, v, *, rt=None, R_nsc=None, a_plu=None, ctx=None):
        return SimpleNamespace(cube={"data": np.ones(shape, dtype=float)})

    def eval_kappa_stub(combined_model_cube, beta, gamma_model_deg, *, ctx=None):
        captured.append(np.array(combined_model_cube, copy=True))
        if len(captured) == 1:
            km._FIT_CTX["disc_cube"] = poisoned_disc_cube
            return np.zeros(1, dtype=float), None
        raise _StopOnSecondLoopEval

    monkeypatch.setattr(km, "build_v_grid_and_label", build_v_grid_stub)
    monkeypatch.setattr(km, "build_model", build_model_stub)
    monkeypatch.setattr(km, "eval_kappa_for_model", eval_kappa_stub)

    with pytest.raises(_StopOnSecondLoopEval):
        km.fit_gridsearch_component(
            obs_for_fit=obs_stub,
            disc_cube=disc_cube_input,
            vel_axis=np.linspace(-50.0, 50.0, shape[0]),
            origin=(1.5, 1.5),
            pixscale=0.2,
            nrebin=1,
            scale=1.0,
            geometry="spherical",
            FIT_MODE="independent",
            gamma_model_deg=45.0,
            aperture_deg=30.0,
            double_cone=False,
            radius_range_model_arcsec=[0.0, 1.0],
            theta_range=[[0.0, 180.0]],
            phi_range=[[0.0, 360.0]],
            zeta_range=[-0.5, 0.5],
            logradius=False,
            psf_sigma=0.0,
            lsf_sigma=0.0,
            vel_sigma=0.0,
            npt=10,
            num_shells=1,
            perc=(0.01, 0.99),
            perc_weights=1.0,
            loss="extreme",
            CRPS_QGRID=None,
            SIGMA_PERC_KMS=20.0,
            beta_min=60.0,
            beta_max=60.0,
            step_beta=1.0,
            v_min=10.0,
            v_max=20.0,
            step_v=10.0,
            verbose_label="Loop regression",
        )

    assert len(captured) == 2
    expected_combined = disc_cube_input + 1.0
    np.testing.assert_allclose(captured[0], expected_combined, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(captured[1], expected_combined, rtol=0.0, atol=0.0)
