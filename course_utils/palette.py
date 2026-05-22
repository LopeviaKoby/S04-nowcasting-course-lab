"""Shared plotting style and radar rain-rate color scales."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


RAIN_LEVELS = [0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0]
RAIN_COLORS = [
    "#eeeeee",  # 0.0 - 0.5: no/trace rain, gray like the reference figure
    "#00a000",  # 0.5 - 2.0: light rain starts green
    "#00bf18",  # 2.0 - 5.0
    "#00e02a",  # 5.0 - 10.0
    "#fff000",  # 10.0 - 20.0
    "#ffb000",  # 20.0 - 30.0
    "#ff5a00",  # 30.0 - 50.0
    "#d60000",  # 50.0 - 60.0
]

ERROR_LEVELS = [0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0]


def rain_cmap():
    """Return a discrete rain-rate colormap and norm in mm/h."""
    cmap = ListedColormap(RAIN_COLORS, name="ideam_rain_rate")
    cmap.set_over("#b30000")
    cmap.set_under("#ffffff")
    norm = BoundaryNorm(RAIN_LEVELS, cmap.N)
    return cmap, norm


def error_cmap():
    """Return a perceptual error colormap and norm in mm/h."""
    cmap = plt.get_cmap("magma").copy()
    norm = BoundaryNorm(ERROR_LEVELS, cmap.N)
    return cmap, norm


def apply_course_style() -> None:
    """Apply a quiet plotting style shared by all notebooks."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titleweight": "regular",
            "font.size": 10,
            "legend.frameon": False,
        }
    )
