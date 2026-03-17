from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from moka3d import moka3d_source as km


def _plot_inputs() -> tuple[SimpleNamespace, SimpleNamespace]:
    flux_data = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    vel_data = np.array(
        [
            [-10.0, 0.0, 10.0],
            [-5.0, 0.0, 5.0],
            [-10.0, 0.0, 10.0],
        ],
        dtype=float,
    )
    sig_data = np.full((3, 3), 20.0, dtype=float)

    obs = SimpleNamespace(
        maps={
            "flux": flux_data,
            "vel": vel_data,
            "sig": sig_data,
        }
    )
    model = SimpleNamespace(
        maps={
            "flux": flux_data * 0.9,
            "vel": vel_data * 0.8,
            "sig": sig_data * 1.1,
        },
        cube={
            "xextent": [-1.5, 1.5],
            "yextent": [-1.5, 1.5],
        },
    )
    return obs, model


def test_plot_kin_maps_3x3_accepts_missing_optional_psf() -> None:
    obs, model = _plot_inputs()
    plt.close("all")

    km.plot_kin_maps_3x3(
        obs=obs,
        m=model,
        xy_AGN=[0.0, 0.0],
        psf_bmaj=None,
        psf_bmin=None,
    )

    fig = plt.gcf()
    ellipse_count = sum(
        isinstance(patch, Ellipse)
        for axis in fig.axes
        for patch in axis.patches
    )
    assert ellipse_count == 0
    plt.close(fig)


def test_plot_kin_maps_3x3_still_draws_psf_overlay_when_inputs_are_provided() -> None:
    obs, model = _plot_inputs()
    plt.close("all")

    km.plot_kin_maps_3x3(
        obs=obs,
        m=model,
        xy_AGN=[0.0, 0.0],
        psf_bmaj=0.2,
        psf_bmin=0.1,
        psf_pa=15.0,
    )

    fig = plt.gcf()
    ellipse_count = sum(
        isinstance(patch, Ellipse)
        for axis in fig.axes
        for patch in axis.patches
    )
    assert ellipse_count > 0
    plt.close(fig)
