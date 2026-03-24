#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:38:31 2026

@author: cosimo
"""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def finalize_figure(output_path: Path | None = None, show: bool = False, close: bool = True) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    if close:
        plt.close()