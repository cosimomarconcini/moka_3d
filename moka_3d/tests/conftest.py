from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
import types

import numpy as np
import pytest

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "moka3d" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

if importlib.util.find_spec("skimage.transform") is None:
    transform_mod = types.ModuleType("skimage.transform")

    def downscale_local_mean(arr: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
        data = np.asarray(arr, dtype=float)
        if data.ndim != len(factors):
            raise ValueError("factors must match the array dimensionality.")

        trimmed = data
        reshape_dims: list[int] = []
        for axis, factor in enumerate(factors):
            factor = int(factor)
            if factor <= 0:
                raise ValueError("Downscale factors must be positive integers.")
            size = trimmed.shape[axis] - (trimmed.shape[axis] % factor)
            sl = [slice(None)] * trimmed.ndim
            sl[axis] = slice(0, size)
            trimmed = trimmed[tuple(sl)]
            reshape_dims.extend([size // factor, factor])

        reshaped = trimmed.reshape(*reshape_dims)
        mean_axes = tuple(range(1, reshaped.ndim, 2))
        return reshaped.mean(axis=mean_axes)

    transform_mod.downscale_local_mean = downscale_local_mean

    skimage_mod = types.ModuleType("skimage")
    skimage_mod.transform = transform_mod

    sys.modules.setdefault("skimage", skimage_mod)
    sys.modules.setdefault("skimage.transform", transform_mod)

if importlib.util.find_spec("rich.progress") is None:
    progress_mod = types.ModuleType("rich.progress")

    class _Column:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class Progress:
        def __init__(self, *args, **kwargs) -> None:
            self._task_id = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def add_task(self, *args, **kwargs) -> int:
            self._task_id += 1
            return self._task_id

        def advance(self, task_id: int, advance: int = 1) -> None:
            return None

    progress_mod.Progress = Progress
    progress_mod.BarColumn = _Column
    progress_mod.TextColumn = _Column
    progress_mod.TimeElapsedColumn = _Column
    progress_mod.TimeRemainingColumn = _Column

    rich_mod = types.ModuleType("rich")
    rich_mod.progress = progress_mod

    sys.modules.setdefault("rich", rich_mod)
    sys.modules.setdefault("rich.progress", progress_mod)

from ._synthetic import (
    SyntheticCubeCase,
    ensure_workspace,
    make_combo_wavelength_case,
    make_disk_frequency_case,
    make_disk_wavelength_case,
    make_ne_map_step,
    make_outflow_wavelength_case,
    make_sn_map_allgood,
    write_fits_image,
)


@pytest.fixture
def synthetic_workspace(tmp_path: Path) -> dict[str, Path]:
    return ensure_workspace(tmp_path)


@pytest.fixture
def sn_map_allgood(synthetic_workspace: dict[str, Path]) -> Path:
    sn_map = make_sn_map_allgood()
    return write_fits_image(synthetic_workspace["ancillary_dir"] / "sn_map_allgood.fits", sn_map)


@pytest.fixture
def ne_map_step(synthetic_workspace: dict[str, Path]) -> Path:
    ne_map = make_ne_map_step()
    return write_fits_image(synthetic_workspace["ancillary_dir"] / "ne_map_step.fits", ne_map)


@pytest.fixture
def fx_disk_wl_small(
    synthetic_workspace: dict[str, Path],
    sn_map_allgood: Path,
) -> SyntheticCubeCase:
    case = make_disk_wavelength_case(
        data_dir=synthetic_workspace["data_dir"],
        ancillary_dir=synthetic_workspace["ancillary_dir"],
    )
    return replace(case, sn_map_path=sn_map_allgood)


@pytest.fixture
def fx_outflow_wl_bicone_small(
    synthetic_workspace: dict[str, Path],
    sn_map_allgood: Path,
) -> SyntheticCubeCase:
    case = make_outflow_wavelength_case(
        data_dir=synthetic_workspace["data_dir"],
        ancillary_dir=synthetic_workspace["ancillary_dir"],
    )
    return replace(case, sn_map_path=sn_map_allgood)


@pytest.fixture
def fx_combo_wl_small(
    synthetic_workspace: dict[str, Path],
    sn_map_allgood: Path,
) -> SyntheticCubeCase:
    case = make_combo_wavelength_case(
        data_dir=synthetic_workspace["data_dir"],
        ancillary_dir=synthetic_workspace["ancillary_dir"],
    )
    return replace(case, sn_map_path=sn_map_allgood)


@pytest.fixture
def fx_disk_freq_small(
    synthetic_workspace: dict[str, Path],
    sn_map_allgood: Path,
) -> SyntheticCubeCase:
    case = make_disk_frequency_case(
        data_dir=synthetic_workspace["data_dir"],
        ancillary_dir=synthetic_workspace["ancillary_dir"],
    )
    return replace(case, sn_map_path=sn_map_allgood)


@pytest.fixture
def fx_outflow_energetics_wl_small(
    fx_outflow_wl_bicone_small: SyntheticCubeCase,
    ne_map_step: Path,
) -> SyntheticCubeCase:
    return replace(
        fx_outflow_wl_bicone_small,
        name="fx_outflow_energetics_wl_small",
        ne_map_path=ne_map_step,
    )


@pytest.fixture
def fx_reporting_reuse(fx_disk_wl_small: SyntheticCubeCase) -> SyntheticCubeCase:
    return fx_disk_wl_small
