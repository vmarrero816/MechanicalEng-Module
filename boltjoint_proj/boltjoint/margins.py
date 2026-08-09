"""Margins of safety for preloaded joints.

Units: in, lbf, psi.

Sign convention: MS > 0 passes, MS < 0 fails.

Key rule (Guideline sec. 11, TM-106943): the factor of safety multiplies the
EXTERNALLY APPLIED loads only.  Preload and thermal loads are not scaled --
they are fixed, known quantities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fasteners import (
    Fastener,
    bolt_thread_shear_area,
    internal_thread_shear_area,
)
from .joint import Joint
from .materials import Material
from .preload import Preload

BASES = ("yield", "ult")

# Recommended separation factors of safety, TM-106943.
SF_SEPARATION_STRUCTURAL = 1.2
SF_SEPARATION_PRESSURE = 1.4


@dataclass(frozen=True)
class MarginResult:
    mode: str
    basis: str
    applied: float          # lbf (or lbf-in for bending)
    allowable: float        # lbf
    SF: float
    MS: float
    note: str = ""

    @property
    def passes(self) -> bool:
        return self.MS >= 0.0


def _ms(allowable: float, applied: float) -> float:
    """MS = allowable/applied - 1, with applied already factored."""
    if applied <= 0.0:
        return math.inf
    return allowable / applied - 1.0


# ---------------------------------------------------------------------------
# Bolt axial load
# ---------------------------------------------------------------------------
def bolt_axial_load(preload: Preload, joint: Joint, P_et: float,
                    SF: float, P_thermal: float = 0.0) -> float:
    """Total axial bolt load, lbf.  TM-106943 eqs. 29-30.

        Pb = Po_max + P_thermal(tension only) + n*phi*SF*P_et

    Uses MAXIMUM preload -- conservative for bolt strength.  Only the tensile
    part of the thermal load is added; a compressive thermal load is ignored
    here and instead applied to the separation check via ``Po_min``.
    """
    return preload.Po_max + max(P_thermal, 0.0) + joint.n * joint.phi * SF * P_et


def load_ratios(P_b: float, V: float, fastener: Fastener, material: Material,
                basis: str, SF_shear: float,
                shear_plane_in_shank: bool = False) -> tuple[float, float]:
    """Return (Rt, Rs) on a single strength basis.

    Rt uses P_b, which must already include the factored external load.
    Rs applies SF_shear to the external shear load.
    """
    Ft, Fs, _ = material.strength(basis)
    P_allow = Ft * fastener.At
    V_allow = Fs * fastener.shear_area(shear_plane_in_shank)
    return P_b / P_allow, SF_shear * V / V_allow


# ---------------------------------------------------------------------------
# Bolt strength modes
# ---------------------------------------------------------------------------
def tension_margin(P_b: float, fastener: Fastener, material: Material,
                   basis: str) -> MarginResult:
    """Bolt tension.  TM-106943 eq. 53.  Uses the TENSILE STRESS AREA."""
    Ft, _, _ = material.strength(basis)
    allow = Ft * fastener.At
    return MarginResult("bolt tension", basis, P_b, allow, 1.0, _ms(allow, P_b),
                        "SF already carried in P_b")


def shear_margin(V: float, fastener: Fastener, material: Material, basis: str,
                 SF: float, shear_plane_in_shank: bool = False) -> MarginResult:
    """Bolt shear.  TM-106943 eqs. 54-55."""
    _, Fs, _ = material.strength(basis)
    A = fastener.shear_area(shear_plane_in_shank)
    allow = Fs * A
    return MarginResult("bolt shear", basis, SF * V, allow, SF, _ms(allow, SF * V),
                        "shank plane" if shear_plane_in_shank else "threads in shear plane")


def combined_margin(Rt: float, Rs: float, basis: str, Rb: float = 0.0,
                    method: str = "tm106943") -> MarginResult:
    """Combined tension / shear (/ bending).

    method='tm106943' : MS = 1/sqrt((Rt+Rb)^2 + Rs^3) - 1     (eqs. 59, 62)
    method='guideline': MOS = 1 - ((Rt+Rb)^2 + Rs^3)          (eq. 51)

    The two do not agree numerically.  The first approximates a true margin;
    the second is a pass/fail indicator.  TM-106943 footnotes that both can be
    unconservative for shear-critical joints -- check the interaction curve
    directly when margins are small.
    """
    interaction = (Rt + Rb) ** 2 + Rs ** 3
    if interaction <= 0.0:
        return MarginResult("combined tension+shear", basis, 0.0, 1.0, 1.0, math.inf, method)
    if method == "tm106943":
        MS = interaction ** -0.5 - 1.0
    elif method == "guideline":
        MS = 1.0 - interaction
    else:
        raise ValueError("method must be 'tm106943' or 'guideline'")
    return MarginResult("combined tension+shear", basis, interaction, 1.0, 1.0, MS,
                        f"{method}; Rt={Rt:.3f} Rs={Rs:.3f} Rb={Rb:.3f}")


def bending_ratio(M: float, fastener: Fastener, material: Material, basis: str,
                  SF: float, rupture_factor: float = 1.0) -> float:
    """Rb = SF*M / bending allowable.  TM-106943 eq. 61.

    The bending allowable is based on the modulus of rupture; ``rupture_factor``
    scales Ftu (1.0 is a conservative default; values above 1 require data).
    """
    Ft, _, _ = material.strength(basis)
    Z = math.pi * fastener.d_minor ** 3 / 32.0     # section modulus, in^3
    allow = rupture_factor * Ft * Z
    return SF * M / allow if allow > 0 else math.inf


# ---------------------------------------------------------------------------
# Joint separation
# ---------------------------------------------------------------------------
def separation_margin(preload: Preload, joint: Joint, P_et: float,
                      SF_sep: float = SF_SEPARATION_STRUCTURAL,
                      P_thermal: float = 0.0) -> MarginResult:
    """Joint separation.  TM-106943 eqs. 67-68.

        P_sep = (1 - n*phi) * P_et
        MS    = Po_min / (SF_sep * P_sep) - 1

    Uses MINIMUM preload.  A preload-relieving (negative) thermal load is
    subtracted from Po_min, per the Guideline sec. 7 convention.
    """
    P_sep = (1.0 - joint.n * joint.phi) * P_et
    Po_eff = preload.Po_min + min(P_thermal, 0.0)
    applied = SF_sep * P_sep
    return MarginResult("joint separation", "n/a", applied, Po_eff, SF_sep,
                        _ms(Po_eff, applied),
                        f"Po_min_eff={Po_eff:.2f} lbf")


# ---------------------------------------------------------------------------
# Thread modes
# ---------------------------------------------------------------------------
def bolt_thread_shear_margin(P_b: float, fastener: Fastener, material: Material,
                             Le: float, basis: str = "ult") -> MarginResult:
    """Bolt external thread shear.  TM-106943 eqs. 63-65."""
    _, Fs, _ = material.strength(basis)
    As = bolt_thread_shear_area(fastener, Le)
    allow = Fs * As
    return MarginResult("bolt thread shear", basis, P_b, allow, 1.0, _ms(allow, P_b),
                        f"As={As:.5f} in^2, Le={Le:.4f} in")


def insert_pullout_margins(P_b: float, fastener: Fastener,
                           bolt_material: Material,
                           insert_material: Material,
                           parent_material: Material,
                           Le_bolt_insert: float,
                           insert_major_dia: float,
                           Le_insert_parent: float,
                           basis: str = "ult") -> list[MarginResult]:
    """The three threaded-insert failure modes.  TM-106943 eqs. 76-79.

    1. insert internal threads strip (bolt pulls out of insert)
    2. insert external threads strip (insert pulls out of parent)
    3. parent internal threads strip

    Each mode uses the correct mating geometry and engagement length -- the
    RevB notebook reused the bolt's minor diameter and Le for all three.
    """
    results: list[MarginResult] = []

    # 1. insert internal thread: mating external thread is the BOLT
    _, Fs, _ = insert_material.strength(basis)
    As = internal_thread_shear_area(fastener.d_major, Le_bolt_insert)
    allow = Fs * As
    results.append(MarginResult("insert internal thread shear", basis, P_b, allow,
                                1.0, _ms(allow, P_b), f"As={As:.5f} in^2"))

    # 2. insert external thread: shear on the insert's own threads
    _, Fs, _ = insert_material.strength(basis)
    As = 5.0 * math.pi * Le_insert_parent * insert_major_dia / 8.0
    allow = Fs * As
    results.append(MarginResult("insert external thread shear", basis, P_b, allow,
                                1.0, _ms(allow, P_b), f"As={As:.5f} in^2"))

    # 3. parent internal thread: mating external thread is the INSERT
    _, Fs, _ = parent_material.strength(basis)
    As = internal_thread_shear_area(insert_major_dia, Le_insert_parent)
    allow = Fs * As
    results.append(MarginResult("parent internal thread shear", basis, P_b, allow,
                                1.0, _ms(allow, P_b), f"As={As:.5f} in^2"))
    return results


def pullout_allowable(results: list[MarginResult]) -> MarginResult:
    """The governing (weakest) pull-out mode.

    Note this takes the MINIMUM allowable.  The RevB notebook used np.max,
    which reported the strongest mode and overstated the margin.
    """
    return min(results, key=lambda r: r.allowable)


# ---------------------------------------------------------------------------
# Abutment modes
# ---------------------------------------------------------------------------
def shear_tearout_margin(V: float, material: Material, thickness: float,
                         edge_distance: float, fastener: Fastener,
                         SF: float, basis: str = "ult") -> MarginResult:
    """Shear tear-out of the sheet edge.  TM-106943 eqs. 69-71.

        As = 2*t*(e - D/2),  MS = Fsu*As/(SF*V) - 1
    """
    _, Fs, _ = material.strength(basis)
    As = 2.0 * thickness * (edge_distance - fastener.d_major / 2.0)
    allow = Fs * As
    eD = edge_distance / fastener.d_major
    note = f"e/D={eD:.2f}"
    if eD < 1.5:
        note += " -- BELOW 1.5, use Bruhn/TM X-73305 method"
    elif eD < 2.0:
        note += " -- below the recommended 2.0"
    return MarginResult("shear tear-out", basis, SF * V, allow, SF,
                        _ms(allow, SF * V), note)


def bearing_margin(V: float, material: Material, thickness: float,
                   fastener: Fastener, SF: float, basis: str = "ult") -> MarginResult:
    """Bolt bearing on the through-hole.  TM-106943 eqs. 72-74."""
    _, _, Fbr = material.strength(basis)
    Abr = fastener.d_major * thickness
    allow = Fbr * Abr
    return MarginResult("bolt bearing", basis, SF * V, allow, SF, _ms(allow, SF * V),
                        f"Abr={Abr:.5f} in^2")


def head_bearing_margin(P_b: float, material: Material, fastener: Fastener,
                        hole_dia: float, washer_od: float | None = None,
                        basis: str = "yield") -> MarginResult:
    """Bearing under the bolt head or washer.  TM-106943 eq. 75.

        Abr = pi/4 * (dh^2 - dt^2)

    Note this is a STRESS check: the RevB notebook computed ``P_b * A``
    instead of ``P_b / A``.
    """
    _, _, Fbr = material.strength(basis)
    # a washer only helps if it is larger than the head
    dh = max(fastener.head_dia, washer_od) if washer_od else fastener.head_dia
    if dh <= hole_dia:
        raise ValueError("bearing diameter must exceed the through-hole diameter")
    Abr = math.pi / 4.0 * (dh ** 2 - hole_dia ** 2)
    allow = Fbr * Abr
    return MarginResult("head/washer bearing", basis, P_b, allow, 1.0, _ms(allow, P_b),
                        f"stress={P_b / Abr:,.0f} psi, Abr={Abr:.5f} in^2")
