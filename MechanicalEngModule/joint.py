"""Joint stiffness, loading-plane factor and thermal load.

Units: in, lbf, psi, degF.

Configurations follow TM-106943 figs. 7-10:
    1 : through-bolt with nut, protruding head              (eqs. 31-35)
    2 : through-bolt with nut, 100 deg flat head            (eqs. 36-41)
    3 : bolt into threaded insert / tapped hole, protruding head (eqs. 42-46)
    4 : bolt into threaded insert / tapped hole, flat head  (eqs. 47-52)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fasteners import Fastener
from .materials import Material


@dataclass
class Layer:
    """One clamped layer."""
    thickness: float          # in
    material: Material

    @property
    def E(self) -> float:
        return self.material.E

    @property
    def alpha(self) -> float:
        return self.material.alpha


@dataclass
class Joint:
    """Preloaded joint geometry and stiffness.

    Parameters
    ----------
    fastener      : Fastener
    layers        : clamped layers, in stack order, head end first.  For
                    configurations 3 and 4 the LAST layer is the tapped parent
                    material and ``insert_length`` must be given.
    bolt_material : Material of the bolt
    configuration : 1..4 (see module docstring)
    insert_length : Li, thread engagement into the insert/parent, in.
                    Required for configurations 3 and 4.
    stiffness_method : 'tm106943' uses the closed-form joint stiffness with an
                    equivalent modulus (TM-106943 eqs. 33/38/44/49).
                    'frustum' builds the stack layer by layer using Shigley
                    cones and combines them in series (Guideline sec. 5.2).
                    The frustum method is preferred for stacks of dissimilar
                    materials and is required to see per-layer behaviour.
    washer_face_dia : small-end diameter of the stress cone, in.  Defaults to
                    the head diameter.  Only used by the frustum method.
    cone_angle_deg : frustum half-apex angle.  Shigley recommends 30 deg.
    """

    fastener: Fastener
    layers: list[Layer]
    bolt_material: Material
    configuration: int = 1
    insert_length: float | None = None
    stiffness_method: str = "tm106943"
    washer_face_dia: float | None = None
    cone_angle_deg: float = 30.0

    L: float = field(init=False)       # effective grip length, in
    kb: float = field(init=False)      # bolt stiffness, lbf/in
    kj: float = field(init=False)      # joint stiffness, lbf/in
    n: float = field(init=False)       # loading plane factor, 0..1
    Ej: float = field(init=False)      # equivalent joint modulus, psi

    def __post_init__(self):
        self._validate()
        self.L = self._grip_length()
        self.Ej = self._equivalent_modulus()
        self.kb = self._bolt_stiffness()
        self.kj = self._joint_stiffness()
        self.n = self._loading_plane_factor()

    # -- validation --------------------------------------------------------
    def _validate(self) -> None:
        if self.configuration not in (1, 2, 3, 4):
            raise ValueError("configuration must be 1, 2, 3 or 4")
        if len(self.layers) < 2:
            raise ValueError("need at least 2 layers")
        if self.configuration in (3, 4):
            if self.insert_length is None:
                raise ValueError("configurations 3 and 4 require insert_length")
            if self.insert_length > self.layers[-1].thickness:
                raise ValueError(
                    "insert_length exceeds parent material thickness "
                    f"({self.insert_length} > {self.layers[-1].thickness})"
                )
        flat = self.fastener.head_style == "flat100"
        if self.configuration in (1, 3) and flat:
            raise ValueError(
                f"configuration {self.configuration} is for protruding heads; "
                "use 2 or 4 for a 100 deg flat head"
            )
        if self.configuration in (2, 4) and not flat:
            raise ValueError(
                f"configuration {self.configuration} requires head_style='flat100'"
            )

    # -- helpers -----------------------------------------------------------
    @property
    def _t(self) -> list[float]:
        return [ly.thickness for ly in self.layers]

    @property
    def total_thickness(self) -> float:
        return sum(self._t)

    def _effective_thicknesses(self) -> list[float]:
        """Per-layer thickness contributing to the grip, per configuration."""
        t = list(self._t)
        if self.configuration in (2, 4):      # countersunk head buries lh/2
            t[0] = t[0] - self.fastener.head_height / 2.0
        if self.configuration in (3, 4):      # engagement buries Li/2
            t[-1] = t[-1] - self.insert_length / 2.0
        if any(x <= 0 for x in t):
            raise ValueError("effective layer thickness became non-positive")
        return t

    def effective_thicknesses(self) -> list[float]:
        """Per-layer thickness contributing to the grip, in."""
        return self._effective_thicknesses()

    def _grip_length(self) -> float:
        return sum(self._effective_thicknesses())

    def _equivalent_modulus(self) -> float:
        """TM-106943 eqs. 34/40/45: Ej = L / sum(li/Ei)."""
        te = self._effective_thicknesses()
        return self.L / sum(ti / ly.E for ti, ly in zip(te, self.layers))

    # -- stiffness ---------------------------------------------------------
    def _bolt_stiffness(self) -> float:
        """Series stiffness of shank and threaded portion, lbf/in.

        If the fastener has no reduced shank the result reduces to At*E/L,
        matching TM-106943 eqs. 32/37/43.
        """
        f, E = self.fastener, self.bolt_material.E
        if abs(f.shank_dia - f.d_major) < 1e-12:
            return f.At * E / self.L
        # crude split: threads occupy the engagement length, shank the rest
        lt = self.insert_length if self.insert_length else f.d_major
        lt = min(lt, self.L)
        ls = self.L - lt
        return 1.0 / (ls / (f.A_shank * E) + lt / (f.At * E))

    def _joint_stiffness(self) -> float:
        """Clamped-material stiffness, lbf/in."""
        if self.stiffness_method == "frustum":
            from .thermal import member_stiffness_frustum
            return member_stiffness_frustum(self, self.washer_face_dia,
                                            self.cone_angle_deg)
        if self.stiffness_method != "tm106943":
            raise ValueError("stiffness_method must be 'tm106943' or 'frustum'")
        D, L, Ej = self.fastener.d_major, self.L, self.Ej
        if self.configuration == 1:                       # eq. 33
            return math.pi * Ej * D / (2.0 * math.log(5.0 * (L + 0.5 * D) / (L + 2.5 * D)))
        if self.configuration == 3:                       # eq. 44
            return math.pi * Ej * D / math.log(5.0 * (2 * L + 0.5 * D) / (2 * L + 2.5 * D))
        # configurations 2 and 4: countersunk head, eqs. 38 / 49
        dw = (self.fastener.head_dia + D) / 2.0
        if dw <= D:
            raise ValueError("head bearing diameter must exceed the major diameter")
        num = (L + dw - D) * (dw + D)
        den = (L + dw + D) * (dw - D)
        return math.pi * Ej * D / math.log(num / den)

    def _loading_plane_factor(self) -> float:
        """TM-106943 eqs. 35/41/46/52.  Dimensionless, 0 <= n <= 1.

        n = (distance between loading planes) / (total joint thickness)
        """
        te = self._effective_thicknesses()
        t = self._t
        first = te[0] / 2.0 if self.configuration in (1, 3) else te[0]
        last = te[-1] / 2.0 if self.configuration in (1, 2) else te[-1]
        middle = sum(te[1:-1])
        n = (first + middle + last) / sum(t)
        return min(max(n, 0.0), 1.0)

    # -- derived -----------------------------------------------------------
    @property
    def phi(self) -> float:
        """Stiffness ratio kb/(kb+kj), dimensionless."""
        return self.kb / (self.kb + self.kj)

    @property
    def alpha_joint_eq(self) -> float:
        """Thickness-weighted CTE of the clamped stack, 1/degF."""
        te = self._effective_thicknesses()
        return sum(ti * ly.alpha for ti, ly in zip(te, self.layers)) / sum(te)

    def thermal_load(self, delta_T: float, method: str | None = None) -> float:
        """Change in bolt axial load from a uniform temperature change, lbf.

        method='tm106943' : P_th = keff * L * dT * (alpha_j_eq - alpha_b)
        method='layered'  : Guideline sec. 7, per-layer expansion and a
                            series-of-frusta member stiffness.

        Defaults to 'layered' when the joint was built with
        stiffness_method='frustum', otherwise 'tm106943'.

        Positive result increases bolt tension; negative relieves preload.
        """
        if method is None:
            method = "layered" if self.stiffness_method == "frustum" else "tm106943"
        if method == "layered":
            from .thermal import thermal_load_layered
            return thermal_load_layered(self, delta_T, self.washer_face_dia,
                                        self.cone_angle_deg).P_thermal
        if method != "tm106943":
            raise ValueError("method must be 'tm106943' or 'layered'")
        keff = self.kb * self.kj / (self.kb + self.kj)
        return keff * self.L * delta_T * (self.alpha_joint_eq - self.bolt_material.alpha)

    def thermal_breakdown(self, delta_T: float):
        """Full per-layer thermal result.  See thermal.ThermalResult."""
        from .thermal import thermal_load_layered
        return thermal_load_layered(self, delta_T, self.washer_face_dia,
                                    self.cone_angle_deg)

    def layer_stiffnesses(self):
        """The frusta making up the clamped stack.  See thermal.Frustum."""
        from .thermal import build_frusta
        return build_frusta(self, self.washer_face_dia, self.cone_angle_deg)

    def summary(self) -> dict[str, float]:
        return {
            "grip length L [in]": self.L,
            "Ej [psi]": self.Ej,
            "kb [lbf/in]": self.kb,
            "kj [lbf/in]": self.kj,
            "phi = kb/(kb+kj)": self.phi,
            "loading plane factor n": self.n,
            "method": self.stiffness_method,
            "alpha_joint_eq [ppm/F]": self.alpha_joint_eq * 1e6,
            "alpha_bolt [ppm/F]": self.bolt_material.alpha * 1e6,
        }
