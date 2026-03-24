#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "moka3d" / "src"
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "moka3d-mpl-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from moka3d import moka3d_source as km


DEFAULT_SHELLS = (1, 8, 16)


def _radius_ranges(n_shells: int, rmin: float = 0.0, rmax: float = 1.6) -> list[list[float]]:
    edges = np.linspace(float(rmin), float(rmax), int(n_shells) + 1, dtype=float)
    return [[float(edges[i]), float(edges[i + 1])] for i in range(int(n_shells))]


def _build_component(n_shells: int, *, npt_total: int) -> object:
    beta_arr = np.linspace(55.0, 65.0, int(n_shells), dtype=float)
    v_arr = np.linspace(120.0, 160.0, int(n_shells), dtype=float)
    model = km._make_multishell_component(
        npt_total=int(npt_total),
        n_shells=int(n_shells),
        geometry="cylindrical",
        radius_range_shells=_radius_ranges(int(n_shells)),
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
        v_arr=v_arr,
        beta_arr=beta_arr,
        xycenter=[0.0, 0.0],
        alpha=0.0,
        gamma=35.0,
        vsys=0.0,
    )

    expected_npt = int(n_shells) * int(int(npt_total) / int(n_shells))
    if int(model.npt) != expected_npt:
        raise AssertionError(f"Unexpected final npt: expected {expected_npt}, got {model.npt}")
    if getattr(model, "geometry", None) != "cylindrical":
        raise AssertionError(f"Unexpected geometry: {getattr(model, 'geometry', None)!r}")
    if not np.isfinite(np.asarray(model.flux, dtype=float)).all():
        raise AssertionError("Model flux contains non-finite values.")

    return model


def _time_case(n_shells: int, *, npt_total: int, repeats: int, warmup: int) -> dict[str, object]:
    for _ in range(int(warmup)):
        _build_component(int(n_shells), npt_total=int(npt_total))

    times_s: list[float] = []
    last_model = None
    for _ in range(int(repeats)):
        gc.collect()
        start = perf_counter()
        last_model = _build_component(int(n_shells), npt_total=int(npt_total))
        times_s.append(perf_counter() - start)

    assert last_model is not None
    return {
        "n_shells": int(n_shells),
        "npt_total_requested": int(npt_total),
        "npt_final": int(last_model.npt),
        "radius_size": int(np.asarray(last_model.radius).size),
        "flux_sum": float(np.sum(np.asarray(last_model.flux, dtype=float))),
        "wall_time_s": times_s,
        "wall_time_min_s": float(min(times_s)),
        "wall_time_median_s": float(np.median(times_s)),
        "wall_time_max_s": float(max(times_s)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the multi-shell assembly cost of moka3d._make_multishell_component()."
    )
    parser.add_argument(
        "--shells",
        nargs="+",
        type=int,
        default=list(DEFAULT_SHELLS),
        help="Shell counts to benchmark.",
    )
    parser.add_argument(
        "--npt-total",
        type=int,
        default=4800,
        help="Total requested cloud count passed into _make_multishell_component().",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per shell count.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of untimed warmup runs per shell count.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON benchmark output.",
    )
    args = parser.parse_args()

    results = [
        _time_case(int(n_shells), npt_total=int(args.npt_total), repeats=int(args.repeats), warmup=int(args.warmup))
        for n_shells in args.shells
    ]
    payload = {
        "benchmark": "_make_multishell_component",
        "config": {
            "shells": [int(x) for x in args.shells],
            "npt_total": int(args.npt_total),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "geometry": "cylindrical",
            "cube_nbins": [21, 13, 13],
        },
        "results": results,
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
