"""Layer-by-layer member stiffness and thermal loads.

Units: in, lbf, psi, degF.

Implements the Shigley frustum method (Guideline sec. 5.2, eq. 18) and the
layer-by-layer thermal formulation (Guideline sec. 7, eqs. 29-33).

Layer-by-layer thermal, derived
-------------------------------
Bolt extension, from free expansion plus the constraint load::

    dL_bolt = alpha_b * L * dT + P_th / k_b

Each layer, compressed by the same force (springs in series)::

    dL_i = alpha_i * l_i * dT - P_th / k_i

Compatibility (eq. 31) requires ``dL_bolt = sum(dL_i)``, so::

    P_th = dT * (sum(alpha_i * l_i) - alpha_b * L) / (1/k_b + 1/k_j)

which is the closed form used below.  Note it collapses to the TM-106943
single-CTE relation when ``sum(alpha_i*l_i)`` is written as ``alpha_eq * L``;
the layer-by-layer part that actually changes the answer is ``k_j`` computed as
a series of per-layer frustum stiffnesses rather than from one closed-form
expression with an equivalent modulus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TAN30 = math.tan(math.radians(30.0))     # 0.57735
COT_FACTOR = 2.0 * TAN30                 # 1.1547, the "1.155" in Shigley eq. 8-20


@dataclass(frozen=True)
class Frustum:
    """One conical stiffness element in the clamped stack."""

    index: int              # source layer index
    thickness: float        # in
    E: float                # psi
    alpha: float            # 1/degF
    d_start: float          # small-end diameter of the cone, in
    k: float                # lbf/in
    x_start: float          # distance from the head bearing face, in

    @property
    def compliance(self) -> float:
        return 1.0 / self.k


def frustum_stiffness(E: float, d_bolt: float, d_start: float, t: float,
                      angle_deg: float = 30.0) -> float:
    """Stiffness of one hollow-cone frustum, lbf/in.  Guideline eq. 18.

        k = pi*E*d*tan(a) / ln( ((2t*tan(a) + D - d)(D + d)) /
                                ((2t*tan(a) + D + d)(D - d)) )

    Parameters
    ----------
    E       : modulus of the layer, psi
    d_bolt  : bolt major diameter, in
    d_start : small-end diameter of this frustum, in.  Must exceed d_bolt.
    t       : frustum thickness, in
    angle_deg : half-apex angle.  Shigley recommends 30 deg with a washer;
                25-33 deg is the defensible range.  45 deg overestimates.
    """
    if t <= 0.0:
        raise ValueError(f"frustum thickness must be positive, got {t}")
    if d_start <= d_bolt:
        raise ValueError(
            f"frustum start diameter {d_start:.4f} must exceed the bolt "
            f"diameter {d_bolt:.4f} in"
        )
    ta = math.tan(math.radians(angle_deg))
    num = (2.0 * t * ta + d_start - d_bolt) * (d_start + d_bolt)
    den = (2.0 * t * ta + d_start + d_bolt) * (d_start - d_bolt)
    return math.pi * E * d_bolt * ta / math.log(num / den)


def build_frusta(joint, washer_face_dia: float | None = None,
                 angle_deg: float = 30.0) -> list[Frustum]:
    """Decompose the clamped stack into frusta and return them in order.

    The stress cones start at both bearing faces and expand toward the joint
    midplane, so the stack is split at every material boundary AND at the
    midplane (Guideline sec. 5.2: symmetric frusta are required for
    equilibrium, and the start diameter of each layer must account for the
    preceding frustum).

    For a threaded hole the far bearing face is taken at the midpoint of the
    engaged threads, which is already how ``Joint`` computes its effective
    thicknesses.
    """
    te = joint.effective_thicknesses()
    L = sum(te)
    mid = L / 2.0
    d_bolt = joint.fastener.d_major
    dw = washer_face_dia if washer_face_dia else joint.fastener.head_dia

    # segment the stack at layer boundaries
    segments: list[tuple[float, float, int]] = []
    x = 0.0
    for i, t in enumerate(te):
        segments.append((x, x + t, i))
        x += t

    # split any segment that straddles the midplane
    split: list[tuple[float, float, int]] = []
    for s, e, i in segments:
        if s < mid - 1e-12 < e - 1e-12 and s < mid < e:
            split.append((s, mid, i))
            split.append((mid, e, i))
        else:
            split.append((s, e, i))

    frusta: list[Frustum] = []
    for s, e, i in split:
        t = e - s
        if t <= 1e-12:
            continue
        # distance from the nearer bearing face to this frustum's small end
        x0 = s if e <= mid + 1e-12 else L - e
        d_start = dw + 2.0 * x0 * math.tan(math.radians(angle_deg))
        layer = joint.layers[i]
        k = frustum_stiffness(layer.E, d_bolt, d_start, t, angle_deg)
        frusta.append(Frustum(index=i, thickness=t, E=layer.E,
                              alpha=layer.alpha, d_start=d_start, k=k,
                              x_start=s))
    return frusta


def member_stiffness_frustum(joint, washer_face_dia: float | None = None,
                             angle_deg: float = 30.0) -> float:
    """Clamped-material stiffness as a series of frusta, lbf/in."""
    frusta = build_frusta(joint, washer_face_dia, angle_deg)
    return 1.0 / sum(f.compliance for f in frusta)


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalResult:
    """Thermal load and its per-layer breakdown."""

    delta_T: float                  # degF
    P_thermal: float                # lbf, + increases bolt tension
    free_expansion_joint: float     # in, sum(alpha_i * l_i * dT)
    free_expansion_bolt: float      # in, alpha_b * L * dT
    mismatch: float                 # in, joint minus bolt
    layers: list[dict]              # per-layer contributions

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.layers)


def thermal_load_layered(joint, delta_T: float,
                         washer_face_dia: float | None = None,
                         angle_deg: float = 30.0) -> ThermalResult:
    """Layer-by-layer thermal load.  Guideline sec. 7.

    Positive ``P_thermal`` increases bolt tension; negative relieves preload.
    """
    te = joint.effective_thicknesses()
    L = sum(te)
    kb = joint.kb
    frusta = build_frusta(joint, washer_face_dia, angle_deg)
    kj = 1.0 / sum(f.compliance for f in frusta)

    free_joint = sum(t * ly.alpha for t, ly in zip(te, joint.layers)) * delta_T
    free_bolt = joint.bolt_material.alpha * L * delta_T
    mismatch = free_joint - free_bolt

    P_th = mismatch / (1.0 / kb + 1.0 / kj)

    rows = []
    for f in frusta:
        rows.append({
            "layer": f.index + 1,
            "material": joint.layers[f.index].material.name,
            "thickness [in]": f.thickness,
            "d_start [in]": f.d_start,
            "E [psi]": f.E,
            "alpha [ppm/F]": f.alpha * 1e6,
            "k_i [lbf/in]": f.k,
            "free expansion [in]": f.thickness * f.alpha * delta_T,
            "constrained defl [in]": -P_th / f.k,
        })
    return ThermalResult(delta_T=delta_T, P_thermal=P_th,
                         free_expansion_joint=free_joint,
                         free_expansion_bolt=free_bolt,
                         mismatch=mismatch, layers=rows)


@dataclass(frozen=True)
class ThermalEnvironment:
    """Assembly, hot and cold temperatures.

    Temperatures in degC by default; set ``celsius=False`` to supply degF.
    Only the differences from assembly matter, so unit offsets cancel.
    """

    T_assembly: float
    T_hot: float
    T_cold: float
    celsius: bool = True

    def __post_init__(self):
        if self.T_hot < self.T_assembly:
            raise ValueError("T_hot must be at or above T_assembly")
        if self.T_cold > self.T_assembly:
            raise ValueError("T_cold must be at or below T_assembly")

    def _delta(self, T: float) -> float:
        d = T - self.T_assembly
        return 1.8 * d if self.celsius else d

    @property
    def cases(self) -> dict[str, float]:
        """Case name -> temperature change in degF."""
        return {
            "ambient": 0.0,
            "hot": self._delta(self.T_hot),
            "cold": self._delta(self.T_cold),
        }
