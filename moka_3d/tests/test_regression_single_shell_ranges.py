from __future__ import annotations

import numpy as np
from moka3d import moka3d_source as km


def test_make_multishell_component_accepts_flat_single_shell_range() -> None:
    model = km._make_multishell_component(
        npt_total=200,
        n_shells=1,
        geometry="cylindrical",
        radius_range_shells=[0.0, 1.0],
        theta_range=[[0, 1]],
        phi_range=[[0, 360]],
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
            np.array([-315.0, 315.0]),
            np.array([-1.3, 1.3]),
            np.array([-1.3, 1.3]),
        ],
        cube_nbins=(21, 13, 13),
        fluxpars=[1.0, 0.05],
        v_arr=[140.0],
        beta_arr=[60.0],
        xycenter=[0.0, 0.0],
        alpha=0.0,
        gamma=35.0,
        vsys=0.0,
    )

    assert model is not None
    assert model.geometry == "cylindrical"
    assert np.allclose(model.radius_range, [0.0, 1.0])
    assert int(model.npt) == 200
    assert getattr(model, "phi").shape == (200,)
