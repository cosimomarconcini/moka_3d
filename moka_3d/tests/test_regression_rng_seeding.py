from __future__ import annotations

import numpy as np
import pytest

from moka3d import moka3d_source as km


_TEST_SEEDS = {
    "theta": 101,
    "zeta": 102,
    "phi": 103,
    "radius": 104,
    "vsigx": 105,
    "vsigy": 106,
    "vsigz": 107,
    "xpsf": 108,
    "ypsf": 109,
    "zlsf": 110,
}


def _make_seeded_model() -> km.model:
    return km.model(
        npt=32,
        use_seeds=True,
        seeds=dict(_TEST_SEEDS),
        geometry="cylindrical",
        logradius=False,
        radius_range=[0.1, 0.9],
        theta_range=[[0.0, 180.0]],
        phi_range=[[15.0, 165.0]],
        zeta_range=[-0.2, 0.2],
        vel_sigma=0.0,
        psf_sigma=0.0,
        lsf_sigma=0.0,
        cube_range=[[-50.0, 50.0], [-1.0, 1.0], [-1.0, 1.0]],
        cube_nbins=[5, 5, 5],
    )


def _make_default_seed_policy_model() -> km.model:
    return km.model(
        npt=32,
        geometry="cylindrical",
        logradius=False,
        radius_range=[0.1, 0.9],
        theta_range=[[0.0, 180.0]],
        phi_range=[[15.0, 165.0]],
        zeta_range=[-0.2, 0.2],
        vel_sigma=0.0,
        psf_sigma=0.0,
        lsf_sigma=0.0,
        cube_range=[[-50.0, 50.0], [-1.0, 1.0], [-1.0, 1.0]],
        cube_nbins=[5, 5, 5],
    )


def _next_global_random_after_model(seed: int) -> float:
    np.random.seed(seed)
    _make_seeded_model()
    return float(np.random.random())


def test_model_constructor_does_not_mutate_numpy_global_rng_state() -> None:
    seed = 24680

    np.random.seed(seed)
    expected_next = float(np.random.random())
    observed_next = _next_global_random_after_model(seed)

    assert observed_next == expected_next


def test_equivalent_explicitly_seeded_model_constructions_remain_identical_for_reproducibility() -> None:
    model_a = _make_seeded_model()
    model_b = _make_seeded_model()

    np.testing.assert_allclose(model_a.radius, model_b.radius, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model_a.phi, model_b.phi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model_a.zeta, model_b.zeta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model_a.xpsf, model_b.xpsf, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model_a.ypsf, model_b.ypsf, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model_a.zlsf, model_b.zlsf, rtol=0.0, atol=0.0)


@pytest.mark.xfail(
    reason=(
        "Deferred RNG-policy change: the constructor no longer mutates NumPy's "
        "global RNG, but the default model path still uses fixed internal seeds "
        "for reproducibility, so repeated default constructions remain identical."
    ),
    strict=True,
)
def test_default_equivalent_model_constructions_should_not_be_forced_identical() -> None:
    model_a = _make_default_seed_policy_model()
    model_b = _make_default_seed_policy_model()

    assert not np.array_equal(model_a.radius, model_b.radius)
