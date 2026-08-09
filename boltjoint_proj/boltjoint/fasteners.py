"""Fastener geometry: unified (inch) thread series.

All dimensions in inches.  Areas in in^2.

References
----------
[1] Chambers, J.A., "Preloaded Joint Analysis Methodology for Space Flight
    Systems," NASA TM-106943, 1995.
[2] "Guideline for Bolted Joint Design and Analysis," Version 1.0 (SAND2008-0371).
[3] Machinery's Handbook, unified screw thread formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Thread table.  key -> (major diameter [in], threads per inch)
# UNC and UNF for common small aerospace sizes.
# ---------------------------------------------------------------------------
THREADS: dict[str, tuple[float, float]] = {
    "0-80": (0.0600, 80),
    "1-64": (0.0730, 64),
    "1-72": (0.0730, 72),
    "2-56": (0.0860, 56),
    "2-64": (0.0860, 64),
    "3-48": (0.0990, 48),
    "3-56": (0.0990, 56),
    "4-40": (0.1120, 40),
    "4-48": (0.1120, 48),
    "5-40": (0.1250, 40),
    "5-44": (0.1250, 44),
    "6-32": (0.1380, 32),
    "6-40": (0.1380, 40),
    "8-32": (0.1640, 32),
    "8-36": (0.1640, 36),
    "10-24": (0.1900, 24),
    "10-32": (0.1900, 32),
    "12-24": (0.2160, 24),
    "12-28": (0.2160, 28),
    "1/4-20": (0.2500, 20),
    "1/4-28": (0.2500, 28),
    "5/16-18": (0.3125, 18),
    "5/16-24": (0.3125, 24),
    "3/8-16": (0.3750, 16),
    "3/8-24": (0.3750, 24),
    "7/16-14": (0.4375, 14),
    "7/16-20": (0.4375, 20),
    "1/2-13": (0.5000, 13),
    "1/2-20": (0.5000, 20),
}

# Socket head cap screw head dimensions, ASME B18.3 (nominal max).
# key -> (head diameter [in], head height [in])
SOCKET_HEAD: dict[str, tuple[float, float]] = {
    "0": (0.096, 0.060),
    "1": (0.118, 0.073),
    "2": (0.140, 0.086),
    "3": (0.161, 0.099),
    "4": (0.183, 0.112),
    "5": (0.205, 0.125),
    "6": (0.226, 0.138),
    "8": (0.270, 0.164),
    "10": (0.312, 0.190),
    "12": (0.343, 0.216),
    "1/4": (0.375, 0.250),
    "5/16": (0.469, 0.3125),
    "3/8": (0.562, 0.375),
    "7/16": (0.656, 0.4375),
    "1/2": (0.750, 0.500),
}

# 100 deg flat head socket cap screw, ASME B18.3 (nominal max head diameter).
FLAT_HEAD_100: dict[str, tuple[float, float]] = {
    "0": (0.138, 0.044),
    "2": (0.172, 0.055),
    "4": (0.225, 0.072),
    "6": (0.279, 0.089),
    "8": (0.332, 0.106),
    "10": (0.385, 0.123),
    "1/4": (0.507, 0.161),
    "5/16": (0.635, 0.198),
    "3/8": (0.762, 0.234),
    "1/2": (1.021, 0.307),
}


def _size_key(designation: str) -> str:
    """'2-56' -> '2';  '1/4-28' -> '1/4'."""
    return designation.rsplit("-", 1)[0]


@dataclass(frozen=True)
class Fastener:
    """Geometry of a single threaded fastener.

    Parameters
    ----------
    designation : e.g. '2-56', '1/4-28'
    head_style  : 'socket' | 'flat100' | 'none'
    shank_dia   : diameter of unthreaded shank, in.  Defaults to major dia.
                  Set explicitly for reduced-shank fasteners.
    high_strength : use the >100 ksi tensile-area formula (Guideline eq. 36).
    """

    designation: str
    head_style: str = "socket"
    shank_dia: float | None = None
    high_strength: bool = False

    d_major: float = field(init=False)
    tpi: float = field(init=False)

    def __post_init__(self):
        if self.designation not in THREADS:
            raise KeyError(
                f"Unknown thread {self.designation!r}. "
                f"Known: {sorted(THREADS)}"
            )
        d, n = THREADS[self.designation]
        object.__setattr__(self, "d_major", d)
        object.__setattr__(self, "tpi", n)
        if self.shank_dia is None:
            object.__setattr__(self, "shank_dia", d)

    # -- basic thread geometry -------------------------------------------
    @property
    def pitch(self) -> float:
        """Thread pitch p = 1/n, in."""
        return 1.0 / self.tpi

    @property
    def d_pitch(self) -> float:
        """Basic pitch diameter, in.  d2 = D - 0.649519*p."""
        return self.d_major - 0.649519 * self.pitch

    @property
    def d_minor(self) -> float:
        """Basic minor diameter, in.  d3 = D - 1.082532*p (= 5*sqrt(3)/8 * p)."""
        return self.d_major - 1.082532 * self.pitch

    # -- areas -------------------------------------------------------------
    @property
    def At(self) -> float:
        """Tensile stress area, in^2.

        NOT rounded.  The RevB notebook applied ``np.round(..., 3)`` here,
        which for a #2-56 turns 0.003696 into 0.004 -- an 8% error.
        """
        if self.high_strength:
            # Guideline eq. 36, using basic pitch dia as an estimate of d_bmp.
            return math.pi * (self.d_pitch / 2.0 - 0.16238 / self.tpi) ** 2
        # Machinery's Handbook / Guideline eq. 35, steels up to 100 ksi Ftu.
        return math.pi / 4.0 * (self.d_major - 0.9743 / self.tpi) ** 2

    @property
    def A_shank(self) -> float:
        """Cross-sectional area of the unthreaded shank, in^2."""
        return math.pi / 4.0 * self.shank_dia ** 2

    def shear_area(self, plane_in_shank: bool = False) -> float:
        """Shear area, in^2.

        TM-106943 (eq. 55 discussion): use the minimum tensile area unless the
        shear plane acts solely on the unthreaded shank.
        """
        return self.A_shank if plane_in_shank else self.At

    # -- head --------------------------------------------------------------
    @property
    def head_dia(self) -> float:
        key = _size_key(self.designation)
        if self.head_style == "socket":
            return SOCKET_HEAD[key][0]
        if self.head_style == "flat100":
            return FLAT_HEAD_100[key][0]
        raise ValueError(f"No head dimensions for head_style={self.head_style!r}")

    @property
    def head_height(self) -> float:
        key = _size_key(self.designation)
        if self.head_style == "socket":
            return SOCKET_HEAD[key][1]
        if self.head_style == "flat100":
            return FLAT_HEAD_100[key][1]
        raise ValueError(f"No head dimensions for head_style={self.head_style!r}")

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Fastener({self.designation!r}, D={self.d_major:.4f} in, "
            f"At={self.At:.5f} in^2)"
        )


# ---------------------------------------------------------------------------
# Thread shear areas
# ---------------------------------------------------------------------------
def bolt_thread_shear_area(fastener: Fastener, Le: float) -> float:
    """External (bolt) thread shear area, in^2.  TM-106943 eq. 63.

    As = 5*pi*Le*D_minor_int / 8

    Uses the basic minor diameter of the mating internal thread.
    """
    return 5.0 * math.pi * Le * fastener.d_minor / 8.0


def internal_thread_shear_area(d_major_ext: float, Le: float) -> float:
    """Internal thread shear area, in^2.  TM-106943 eq. 76.

    As = 3*pi*Le*D_major_ext / 4

    Applies to the internal threads of an insert (mating the bolt) or of the
    parent material (mating the insert's external thread).  Pass the MAJOR
    diameter of the mating EXTERNAL thread and the engagement length of that
    particular thread pair.
    """
    return 3.0 * math.pi * Le * d_major_ext / 4.0


def min_engagement_equal_strength(fastener: Fastener) -> float:
    """Minimum thread engagement for equal-strength threads, in.

    Guideline eq. 34, using basic thread dimensions as an estimate.
    """
    d_mt = fastener.d_minor          # max minor dia of internal thread ~ basic
    d_bmp = fastener.d_pitch         # min pitch dia of external thread ~ basic
    denom = math.pi * d_mt * (0.5 + fastener.tpi * (d_bmp - d_mt) * math.tan(math.radians(30.0)))
    return 2.0 * fastener.At / denom
