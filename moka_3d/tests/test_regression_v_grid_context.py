from __future__ import annotations

import numpy as np
import pytest

from moka3d import moka3d_source as km


@pytest.mark.parametrize(
    ("geometry", "fit_mode", "v_min", "v_max", "step_v", "n_geom_v"),
    [
        ("cylindrical", "disk_kepler", 1.0e6, 1.0e8, 1.0, 7),
        ("spherical", "independent", 50.0, 110.0, 20.0, 99),
    ],
)
def test_build_v_grid_and_label_explicit_inputs_ignore_fit_ctx_poisoning(
    geometry: str,
    fit_mode: str,
    v_min: float,
    v_max: float,
    step_v: float,
    n_geom_v: int,
) -> None:
    km.set_fit_context(
        v_min=v_min,
        v_max=v_max,
        step_v=step_v,
        n_geom_v=n_geom_v,
    )
    legacy_v_array, legacy_rt_array, legacy_label = km.build_v_grid_and_label(geometry, fit_mode)

    km.set_fit_context(
        v_min=-999.0,
        v_max=-111.0,
        step_v=3.0,
        n_geom_v=3,
    )
    explicit_v_array, explicit_rt_array, explicit_label = km.build_v_grid_and_label(
        geometry,
        fit_mode,
        v_min=v_min,
        v_max=v_max,
        step_v=step_v,
        n_geom_v=n_geom_v,
    )

    np.testing.assert_allclose(explicit_v_array, legacy_v_array, rtol=0.0, atol=0.0)
    assert explicit_rt_array == legacy_rt_array
    assert explicit_label == legacy_label
