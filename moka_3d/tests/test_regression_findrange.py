from __future__ import annotations

import numpy as np
import pytest

from moka3d import moka3d_source as km


def test_findrange_accepts_valid_fine_uniform_axis() -> None:
    axis = np.array([5000.0010, 5000.0015, 5000.0020, 5000.0025], dtype=float)

    result = km.utils().findrange(axis)

    np.testing.assert_allclose(
        result,
        np.array([5000.00075, 5000.00275], dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_findrange_raises_value_error_for_non_uniform_axis() -> None:
    axis = np.array([0.0, 0.1, 0.21, 0.30], dtype=float)

    with pytest.raises(ValueError, match="uniform"):
        km.utils().findrange(axis)
