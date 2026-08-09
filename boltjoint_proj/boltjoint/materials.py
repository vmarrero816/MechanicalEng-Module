"""Material allowables.

Units: psi, 1/degF, degF.

WARNING
-------
The values below are TYPICAL room-temperature properties for convenience only.
They are NOT a substitute for MMPDS / MIL-HDBK-5, the applicable material
specification, or the fastener procurement spec.  Verify every value used in
a real analysis and record the source.  Fastener-grade properties differ
substantially from bar/plate properties of the same alloy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Material allowables.

    Parameters
    ----------
    name : identifier
    E    : Young's modulus, psi
    Ftu  : ultimate tensile strength, psi
    Fty  : tensile yield strength, psi
    alpha: coefficient of thermal expansion, 1/degF
    Fsu  : ultimate shear strength, psi.  If None, estimated from Ftu using
           ``shear_factor`` (Bickford / Guideline sec. 11: 0.55 for stainless
           and aluminum, 0.60 for carbon steels).
    Fsy  : shear yield strength, psi.  If None, taken as 0.577*Fty
           (TM-106943, von Mises).
    Fbru / Fbry : bearing allowables, psi.  If None, estimated as 1.5*Ftu /
           1.5*Fty, a common lug-analysis placeholder -- replace with real data.
    kind : 'metal' or 'dielectric'; controls the clamped-material stress check.
    source : free text, propagated into reports.
    """

    name: str
    E: float
    Ftu: float
    Fty: float
    alpha: float
    Fsu: float | None = None
    Fsy: float | None = None
    Fbru: float | None = None
    Fbry: float | None = None
    shear_factor: float = 0.55
    kind: str = "metal"
    source: str = "typical value - VERIFY"

    def __post_init__(self):
        if self.Fsu is None:
            object.__setattr__(self, "Fsu", self.shear_factor * self.Ftu)
        if self.Fsy is None:
            object.__setattr__(self, "Fsy", 0.577 * self.Fty)
        if self.Fbru is None:
            object.__setattr__(self, "Fbru", 1.5 * self.Ftu)
        if self.Fbry is None:
            object.__setattr__(self, "Fbry", 1.5 * self.Fty)

    def strength(self, basis: str) -> tuple[float, float, float]:
        """Return (F_tension, F_shear, F_bearing) for basis 'yield' or 'ult'."""
        if basis == "ult":
            return self.Ftu, self.Fsu, self.Fbru
        if basis == "yield":
            return self.Fty, self.Fsy, self.Fbry
        raise ValueError("basis must be 'yield' or 'ult'")


# ---------------------------------------------------------------------------
# Library.  VERIFY BEFORE USE.
# ---------------------------------------------------------------------------
LIBRARY: dict[str, Material] = {
    "A286": Material(
        name="A286 fastener (AMS 5731/5737, 160 ksi class)",
        E=29.1e6, Ftu=160e3, Fty=120e3, alpha=9.17e-6,
        shear_factor=0.55, source="typical fastener grade - VERIFY",
    ),
    "CRES-A286-annealed": Material(
        name="A286 bar, solution treated + aged",
        E=29.1e6, Ftu=130e3, Fty=85e3, alpha=9.17e-6, shear_factor=0.55,
    ),
    "CRES304": Material(
        name="CRES 304 annealed (300 series)",
        E=28.0e6, Ftu=75e3, Fty=30e3, alpha=9.6e-6, shear_factor=0.55,
        source="annealed bar - fastener grades are higher, VERIFY",
    ),
    "CRES-A286-fastener": Material(
        name="A286 NAS1352 socket cap",
        E=29.1e6, Ftu=160e3, Fty=120e3, alpha=9.17e-6, shear_factor=0.55,
    ),
    "Ti-6Al-4V": Material(
        name="Ti-6Al-4V annealed",
        E=16.5e6, Ftu=130e3, Fty=120e3, alpha=4.9e-6, shear_factor=0.55,
    ),
    "AL6061-T6": Material(
        name="Aluminum 6061-T6",
        E=9.9e6, Ftu=42e3, Fty=35e3, alpha=13.1e-6, shear_factor=0.55,
    ),
    "AL7075-T73": Material(
        name="Aluminum 7075-T73",
        E=10.3e6, Ftu=68e3, Fty=56e3, alpha=12.9e-6, shear_factor=0.55,
    ),
    "AL2024-T4": Material(
        name="Aluminum 2024-T4",
        E=10.6e6, Ftu=64e3, Fty=42e3, alpha=12.9e-6, shear_factor=0.55,
    ),
    "RO4003C": Material(
        name="Rogers RO4003C laminate (out-of-plane)",
        E=0.4e6, Ftu=5.0e3, Fty=5.0e3, alpha=25.0e-6,
        shear_factor=0.55, kind="dielectric",
        source="placeholder - use datasheet Z-axis values",
    ),
}


def get(name: str) -> Material:
    if name not in LIBRARY:
        raise KeyError(f"Unknown material {name!r}. Known: {sorted(LIBRARY)}")
    return LIBRARY[name]
