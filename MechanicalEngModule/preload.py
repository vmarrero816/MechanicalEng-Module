"""Preload, torque and preload bounds.

Units: in, lbf, lbf-in.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fasteners import Fastener
from .materials import Material

# Nut factor K (TM-106943 / 540-PG-8072.1.2).  Verify against the applicable
# lubrication and locking-feature callout.
NUT_FACTOR: dict[str, float] = {
    "dry": 0.20,
    "lubricated": 0.15,
    "cadmium plated": 0.16,
    "arathane": 0.18,
    "solid film lube": 0.12,
}

# Preload uncertainty u (TM-106943 table 2).
UNCERTAINTY: dict[str, float] = {
    "torque wrench, unlubricated": 0.35,
    "torque wrench, lubricated": 0.25,
    "preload indicating washer": 0.10,
    "ultrasonic": 0.05,
    "bolt elongation": 0.05,
    "strain gage": 0.05,
}


@dataclass(frozen=True)
class Preload:
    """Preload state of one fastener.

    Parameters
    ----------
    Po       : nominal preload, lbf
    u        : preload uncertainty, fraction (e.g. 0.35)
    relax    : short-term relaxation, fraction of nominal preload (typ. 0.02-0.05)
    K        : nut factor
    d_major  : fastener major diameter, in
    """

    Po: float
    u: float
    relax: float
    K: float
    d_major: float

    @property
    def Po_max(self) -> float:
        """Maximum expected preload, lbf.  TM-106943: Po*(1+u).

        Relaxation is deliberately NOT applied here -- it would be
        unconservative for the bolt strength check.
        """
        return self.Po * (1.0 + self.u)

    @property
    def P_relax(self) -> float:
        """Short-term preload loss, lbf."""
        return self.relax * self.Po

    @property
    def Po_min(self) -> float:
        """Minimum expected preload, lbf.  TM-106943: Po*(1-u) - P_relax."""
        return self.Po * (1.0 - self.u) - self.P_relax

    @property
    def torque_nominal(self) -> float:
        """T = K*D*Po, lbf-in."""
        return self.K * self.d_major * self.Po

    def torque_range(self) -> tuple[float, float]:
        """Torque bounds corresponding to (Po_min_no_relax, Po_max), lbf-in."""
        lo = self.K * self.d_major * self.Po * (1.0 - self.u)
        return lo, self.K * self.d_major * self.Po_max

    def summary(self) -> dict[str, float]:
        return {
            "Po nominal [lbf]": self.Po,
            "Po_max [lbf]": self.Po_max,
            "Po_min [lbf]": self.Po_min,
            "P_relax [lbf]": self.P_relax,
            "T nominal [lbf-in]": self.torque_nominal,
        }


def preload_from_torque(T: float, K: float, d_major: float,
                        u: float = 0.35, relax: float = 0.05) -> Preload:
    """Build a Preload from an applied torque.  Po = T/(K*D)."""
    return Preload(Po=T / (K * d_major), u=u, relax=relax, K=K, d_major=d_major)


def preload_from_yield_fraction(fastener: Fastener, material: Material,
                                fraction: float = 0.65, K: float = 0.20,
                                u: float = 0.35, relax: float = 0.05) -> Preload:
    """Preload as a fraction of bolt tensile yield.  Po = f*Fty*At.

    Typical f is 0.65 (0.60 for very small fasteners where relaxation and
    torque scatter dominate).
    """
    Po = fraction * material.Fty * fastener.At
    return Preload(Po=Po, u=u, relax=relax, K=K, d_major=fastener.d_major)


def preload_target(Po: float, fastener: Fastener, K: float = 0.20,
                   u: float = 0.35, relax: float = 0.05) -> Preload:
    """Preload from a directly specified target force, lbf."""
    return Preload(Po=Po, u=u, relax=relax, K=K, d_major=fastener.d_major)
