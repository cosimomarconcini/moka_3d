from __future__ import annotations

import numpy as np

from moka3d import moka3d_source as km


def test_make_multishell_component_matches_legacy_iterative_add_model() -> None:
    n_shells = 3
    npt_total = 360
    radius_range_shells = [[0.0, 0.4], [0.4, 0.8], [0.8, 1.2]]
    beta_arr = [55.0, 60.0, 65.0]
    v_arr = [120.0, 140.0, 160.0]
    common = dict(
        geometry="cylindrical",
        theta_range=[[0.0, 1.0]],
        phi_range=[[0.0, 360.0]],
        zeta_range=[-0.5, 0.5],
        logradius=False,
        flux_func=None,
        vel1_func=km.vout,
        vel2_func=km.vout,
        vel3_func=km.vout,
        vel_sigma=0.0,
        psf_sigma=1.0,
        lsf_sigma=30.0,
        cube_range=[
            np.array([-315.0, 315.0], dtype=float),
            np.array([-1.3, 1.3], dtype=float),
            np.array([-1.3, 1.3], dtype=float),
        ],
        cube_nbins=(21, 13, 13),
        fluxpars=[1.0, 0.05],
        xycenter=[0.0, 0.0],
        alpha=0.0,
        gamma=35.0,
        vsys=0.0,
    )

    optimized = km._make_multishell_component(
        npt_total=npt_total,
        n_shells=n_shells,
        radius_range_shells=radius_range_shells,
        v_arr=v_arr,
        beta_arr=beta_arr,
        **common,
    )

    npt_shell = int(npt_total / n_shells)
    legacy = km._make_single_km_component(
        npt=npt_shell,
        radius_range=radius_range_shells[0],
        v=v_arr[0],
        beta=beta_arr[0],
        **common,
    )
    for i in range(1, n_shells):
        shell = km._make_single_km_component(
            npt=npt_shell,
            radius_range=radius_range_shells[i],
            v=v_arr[i],
            beta=beta_arr[i],
            **common,
        )
        legacy.add_model(shell)

    attrs = [
        "radius",
        "theta",
        "zeta",
        "phi",
        "flux",
        "x",
        "y",
        "z",
        "xpsf",
        "ypsf",
        "zlsf",
        "velx",
        "vely",
        "velz",
        "vsigx",
        "vsigy",
        "vsigz",
        "xobs",
        "yobs",
        "zobs",
        "xobs_psf",
        "yobs_psf",
        "vlos_lsf",
        "vlos",
    ]
    for attr in attrs:
        np.testing.assert_allclose(
            np.asarray(getattr(optimized, attr)),
            np.asarray(getattr(legacy, attr)),
            rtol=0.0,
            atol=0.0,
        )

    np.testing.assert_allclose(np.asarray(optimized.Pref), np.asarray(legacy.Pref), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(optimized.Vref), np.asarray(legacy.Vref), rtol=0.0, atol=0.0)
    assert int(optimized.npt) == int(legacy.npt) == npt_total
    assert optimized.geometry == legacy.geometry == "cylindrical"
