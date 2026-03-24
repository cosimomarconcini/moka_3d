from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from moka3d import moka3d_source as km

from ._synthetic import (
    DEFAULT_PIXEL_SCALE_ARCSEC,
    SyntheticCubeCase,
)


_TEST_SEEDS = {
    "theta": 11,
    "zeta": 12,
    "phi": 13,
    "radius": 14,
    "vsigx": 15,
    "vsigy": 16,
    "vsigz": 17,
    "xpsf": 18,
    "ypsf": 19,
    "zlsf": 20,
}


def _obs_stub_from_case(case: SyntheticCubeCase) -> SimpleNamespace:
    cube = np.asarray(fits.getdata(case.cube_path), dtype=float)
    vel_axis = np.asarray(case.expected["velocity_axis_kms"], dtype=float)
    dv = float(vel_axis[1] - vel_axis[0])
    ny, nx = cube.shape[1:]
    pix = float(DEFAULT_PIXEL_SCALE_ARCSEC)
    center_x, center_y = case.center_xy
    xrange = [
        (0 - center_x) * pix - pix / 2.0,
        ((nx - 1) - center_x) * pix + pix / 2.0,
    ]
    yrange = [
        (0 - center_y) * pix - pix / 2.0,
        ((ny - 1) - center_y) * pix + pix / 2.0,
    ]
    vrange = [float(vel_axis[0] - dv / 2.0), float(vel_axis[-1] + dv / 2.0)]
    return SimpleNamespace(
        cube={
            "data": cube,
            "range": [vrange, yrange, xrange],
            "nbins": cube.shape,
        }
    )


def test_make_mod_explicit_model_context_matches_legacy_behavior(
    fx_disk_wl_small: SyntheticCubeCase,
) -> None:
    obs_stub = _obs_stub_from_case(fx_disk_wl_small)
    beta_deg = float(fx_disk_wl_small.expected["disk_inc_deg"])
    vparam = float(fx_disk_wl_small.expected["disk_vmax_kms"])

    model_ctx = km.FitModelContext(
        obs=obs_stub,
        geometry="cylindrical",
        fit_mode="disk_arctan",
        gamma_model=float(fx_disk_wl_small.expected["disk_pa_deg"]),
        xy_agn=[0.0, 0.0],
        radius_range_model=[0.0, 1.0],
        theta_range=[[0.0, 180.0]],
        phi_range=[[0.0, 360.0]],
        zeta_range=[-0.5, 0.5],
        logradius=False,
        vel_sigma=0.0,
        psf_sigma=0.0,
        lsf_sigma=0.0,
        use_seeds=True,
        seeds=dict(_TEST_SEEDS),
        npt=200,
        scale=1.0,
        rt_arcsec=1.1,
        r_nsc_default=5.0,
        a_plu_default=4.0,
    )

    km.set_fit_context(
        obs=obs_stub,
        geometry="cylindrical",
        FIT_MODE="disk_arctan",
        gamma_model=float(fx_disk_wl_small.expected["disk_pa_deg"]),
        xy_AGN=[0.0, 0.0],
        radius_range_model=[0.0, 1.0],
        theta_range=[[0.0, 180.0]],
        phi_range=[[0.0, 360.0]],
        zeta_range=[-0.5, 0.5],
        logradius=False,
        vel_sigma=0.0,
        psf_sigma=0.0,
        lsf_sigma=0.0,
        use_seeds=True,
        seeds=dict(_TEST_SEEDS),
        npt=200,
        scale=1.0,
        RT_ARCSEC=1.1,
        R_nsc_default=5.0,
        a_plu_default=4.0,
    )
    legacy_model = km.make_mod(beta_deg, vparam)

    km.set_fit_context(
        obs=SimpleNamespace(cube={"data": np.zeros_like(obs_stub.cube["data"]), "range": [[-1.0, 1.0], [-0.5, 0.5], [-0.5, 0.5]], "nbins": obs_stub.cube["data"].shape}),
        geometry="spherical",
        FIT_MODE="independent",
        gamma_model=0.0,
        xy_AGN=[0.4, -0.2],
        radius_range_model=[0.0, 0.3],
        theta_range=[[15.0, 75.0]],
        phi_range=[[20.0, 120.0]],
        zeta_range=[-0.1, 0.1],
        logradius=True,
        vel_sigma=12.0,
        psf_sigma=1.5,
        lsf_sigma=5.0,
        use_seeds=True,
        seeds={k: v + 100 for k, v in _TEST_SEEDS.items()},
        npt=30,
        scale=9.0,
        RT_ARCSEC=9.9,
        R_nsc_default=9.0,
        a_plu_default=8.0,
    )
    explicit_model = km.make_mod(beta_deg, vparam, ctx=model_ctx)

    np.testing.assert_allclose(explicit_model.cube["data"], legacy_model.cube["data"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(explicit_model.maps["flux"], legacy_model.maps["flux"], rtol=0.0, atol=0.0, equal_nan=True)
    assert explicit_model.cube["data"].shape == legacy_model.cube["data"].shape
    assert np.isfinite(explicit_model.cube["data"]).any()


def test_build_model_explicit_model_context_matches_legacy_behavior(
    fx_disk_wl_small: SyntheticCubeCase,
) -> None:
    obs_stub = _obs_stub_from_case(fx_disk_wl_small)
    beta_deg = float(fx_disk_wl_small.expected["disk_inc_deg"])
    vparam = float(fx_disk_wl_small.expected["disk_vmax_kms"])

    model_ctx = km.FitModelContext(
        obs=obs_stub,
        geometry="cylindrical",
        fit_mode="independent",
        gamma_model=float(fx_disk_wl_small.expected["disk_pa_deg"]),
        xy_agn=[0.0, 0.0],
        radius_range_model=[0.0, 1.0],
        theta_range=[[0.0, 180.0]],
        phi_range=[[0.0, 360.0]],
        zeta_range=[-0.5, 0.5],
        logradius=False,
        vel_sigma=0.0,
        psf_sigma=0.0,
        lsf_sigma=0.0,
        use_seeds=True,
        seeds=dict(_TEST_SEEDS),
        npt=200,
        scale=1.0,
        rt_arcsec=1.1,
        r_nsc_default=5.0,
        a_plu_default=4.0,
    )

    km.set_fit_context(
        obs=obs_stub,
        geometry="cylindrical",
        FIT_MODE="independent",
        gamma_model=float(fx_disk_wl_small.expected["disk_pa_deg"]),
        xy_AGN=[0.0, 0.0],
        radius_range_model=[0.0, 1.0],
        theta_range=[[0.0, 180.0]],
        phi_range=[[0.0, 360.0]],
        zeta_range=[-0.5, 0.5],
        logradius=False,
        vel_sigma=0.0,
        psf_sigma=0.0,
        lsf_sigma=0.0,
        use_seeds=True,
        seeds=dict(_TEST_SEEDS),
        npt=200,
        scale=1.0,
        RT_ARCSEC=1.1,
        R_nsc_default=5.0,
        a_plu_default=4.0,
    )
    legacy_model = km.build_model(beta_deg, vparam)

    km.set_fit_context(
        obs=SimpleNamespace(cube={"data": np.zeros_like(obs_stub.cube["data"]), "range": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]], "nbins": obs_stub.cube["data"].shape}),
        geometry="spherical",
        FIT_MODE="disk_kepler",
        gamma_model=180.0,
        xy_AGN=[1.0, 1.0],
        radius_range_model=[0.2, 0.4],
        theta_range=[[45.0, 90.0]],
        phi_range=[[90.0, 180.0]],
        zeta_range=[-0.2, 0.2],
        logradius=True,
        vel_sigma=25.0,
        psf_sigma=2.5,
        lsf_sigma=3.0,
        use_seeds=True,
        seeds={k: v + 200 for k, v in _TEST_SEEDS.items()},
        npt=25,
        scale=7.0,
        RT_ARCSEC=3.3,
        R_nsc_default=6.0,
        a_plu_default=7.0,
    )
    explicit_model = km.build_model(beta_deg, vparam, ctx=model_ctx)

    np.testing.assert_allclose(explicit_model.cube["data"], legacy_model.cube["data"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(explicit_model.maps["flux"], legacy_model.maps["flux"], rtol=0.0, atol=0.0, equal_nan=True)
    assert explicit_model.cube["data"].shape == legacy_model.cube["data"].shape
    assert np.isfinite(explicit_model.cube["data"]).any()
