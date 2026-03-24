from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from moka3d import moka3d_source as km

from ._synthetic import SyntheticCubeCase


def test_eval_kappa_explicit_scoring_context_matches_legacy_behavior(
    fx_disk_wl_small: SyntheticCubeCase,
) -> None:
    cube_obs = np.asarray(fits.getdata(fx_disk_wl_small.cube_path), dtype=float)
    vel_axis = np.asarray(fx_disk_wl_small.expected["velocity_axis_kms"], dtype=float)
    beta_deg = float(fx_disk_wl_small.expected["disk_inc_deg"])
    pa_deg = float(fx_disk_wl_small.expected["disk_pa_deg"])
    obs_stub = SimpleNamespace(cube={"data": cube_obs})

    km.set_fit_context(
        obs=obs_stub,
        vel_axis=vel_axis,
        origin=fx_disk_wl_small.center_xy,
        num_shells=3,
        rin_pix=0.0,
        rout_pix=5.0,
        aperture=None,
        double_cone=False,
        SIGMA_PERC_KMS=20.0,
        perc=(0.01, 0.99),
        perc_weights=1.0,
        loss="extreme",
        CRPS_QGRID=None,
        geometry="cylindrical",
        FIT_MODE="independent",
        KEPLER_DEPROJECT=False,
    )

    legacy_kappa, legacy_pack = km.eval_kappa_for_model(cube_obs, beta_deg, pa_deg)

    scoring_ctx = km.FitScoringContext(
        cube_obs=cube_obs,
        vel_axis=vel_axis,
        origin=fx_disk_wl_small.center_xy,
        num_shells=3,
        rin_pix=0.0,
        rout_pix=5.0,
        aperture=None,
        double_cone=False,
        sigma_perc_kms=20.0,
        perc=(0.01, 0.99),
        perc_weights=1.0,
        loss="extreme",
        crps_qgrid=None,
        geometry="cylindrical",
        fit_mode="independent",
        kepler_deproject=False,
    )

    km.set_fit_context(
        obs=SimpleNamespace(cube={"data": np.zeros_like(cube_obs)}),
        vel_axis=np.linspace(-1.0, 1.0, vel_axis.size),
        origin=(0, 0),
        num_shells=1,
        rin_pix=0.0,
        rout_pix=1.0,
        aperture=45.0,
        double_cone=True,
        SIGMA_PERC_KMS=99.0,
        perc=(0.5,),
        perc_weights=2.0,
        loss="crps",
        CRPS_QGRID=np.array([0.5]),
        geometry="spherical",
        FIT_MODE="disk_kepler",
        KEPLER_DEPROJECT=True,
    )

    explicit_kappa, explicit_pack = km.eval_kappa_for_model(cube_obs, beta_deg, pa_deg, ctx=scoring_ctx)

    np.testing.assert_allclose(explicit_kappa, legacy_kappa, rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(explicit_pack[0], legacy_pack[0], rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(explicit_pack[1], legacy_pack[1], rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(explicit_pack[2], legacy_pack[2], rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_array_equal(explicit_pack[4], legacy_pack[4])
    np.testing.assert_allclose(explicit_pack[5], legacy_pack[5], rtol=0.0, atol=0.0, equal_nan=True)

    assert len(explicit_pack[3]) == len(legacy_pack[3])
    for explicit_mask, legacy_mask in zip(explicit_pack[3], legacy_pack[3]):
        np.testing.assert_array_equal(explicit_mask, legacy_mask)
