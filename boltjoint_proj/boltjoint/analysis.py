"""End-to-end joint analysis: runs every failure mode on both strength bases.

Units: in, lbf, psi, degF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import margins as M
from .fasteners import Fastener
from .joint import Joint
from .materials import Material
from .preload import Preload


@dataclass
class LoadCase:
    """Externally applied loads and factors of safety for one case.

    P_et      : external axial (tensile) load per fastener, lbf
    V         : external shear load per fastener, lbf
    Mb        : external bending moment per fastener, lbf-in
    delta_T   : temperature change from assembly, degF
    SF_tension / SF_shear : ultimate or yield factors of safety on the
                EXTERNAL loads only
    SF_separation : 1.2 structural, 1.4 pressure systems
    """

    name: str = "case 1"
    P_et: float = 0.0
    V: float = 0.0
    Mb: float = 0.0
    delta_T: float = 0.0
    SF_tension: float = 1.4
    SF_shear: float = 1.4
    SF_separation: float = M.SF_SEPARATION_STRUCTURAL


@dataclass
class InsertSpec:
    """Threaded insert geometry and materials."""
    insert_material: Material
    parent_material: Material
    insert_major_dia: float          # major dia of the insert's EXTERNAL thread, in
    Le_bolt_insert: float            # bolt-into-insert engagement, in
    Le_insert_parent: float          # insert-into-parent engagement, in


@dataclass
class AbutmentSpec:
    """Sheet/lug geometry for tear-out and bearing checks."""
    material: Material
    thickness: float                 # in
    edge_distance: float             # hole centre to free edge, in
    hole_dia: float                  # through-hole diameter, in
    washer_od: float | None = None   # in


@dataclass
class JointAnalysis:
    """Runs the full margin set for one joint and one load case."""

    fastener: Fastener
    bolt_material: Material
    joint: Joint
    preload: Preload
    load_case: LoadCase
    insert: InsertSpec | None = None
    abutment: AbutmentSpec | None = None
    shear_plane_in_shank: bool = False
    combined_method: str = "tm106943"

    P_thermal: float = field(init=False)
    P_b: float = field(init=False)
    results: list[M.MarginResult] = field(init=False, default_factory=list)

    def __post_init__(self):
        lc = self.load_case
        self.P_thermal = self.joint.thermal_load(lc.delta_T)
        self.P_b = M.bolt_axial_load(self.preload, self.joint, lc.P_et,
                                     lc.SF_tension, self.P_thermal)
        self.results = self._run()

    # ------------------------------------------------------------------
    def _run(self) -> list[M.MarginResult]:
        lc, f, mat = self.load_case, self.fastener, self.bolt_material
        out: list[M.MarginResult] = []

        for basis in M.BASES:
            out.append(M.tension_margin(self.P_b, f, mat, basis))
            out.append(M.shear_margin(lc.V, f, mat, basis, lc.SF_shear,
                                      self.shear_plane_in_shank))
            Rt, Rs = M.load_ratios(self.P_b, lc.V, f, mat, basis, lc.SF_shear,
                                   self.shear_plane_in_shank)
            Rb = (M.bending_ratio(lc.Mb, f, mat, basis, lc.SF_tension)
                  if lc.Mb else 0.0)
            out.append(M.combined_margin(Rt, Rs, basis, Rb, self.combined_method))

        out.append(M.separation_margin(self.preload, self.joint, lc.P_et,
                                       lc.SF_separation, self.P_thermal))

        Le = (self.insert.Le_bolt_insert if self.insert
              else (self.joint.insert_length or f.d_major))
        out.append(M.bolt_thread_shear_margin(self.P_b, f, mat, Le, "ult"))

        if self.insert:
            ins = self.insert
            out.extend(M.insert_pullout_margins(
                self.P_b, f, mat, ins.insert_material, ins.parent_material,
                ins.Le_bolt_insert, ins.insert_major_dia, ins.Le_insert_parent,
                "ult"))

        if self.abutment:
            ab = self.abutment
            for basis in M.BASES:
                out.append(M.shear_tearout_margin(lc.V, ab.material, ab.thickness,
                                                  ab.edge_distance, f, lc.SF_shear, basis))
                out.append(M.bearing_margin(lc.V, ab.material, ab.thickness, f,
                                            lc.SF_shear, basis))
            out.append(M.head_bearing_margin(self.P_b, ab.material, f,
                                             ab.hole_dia, ab.washer_od, "yield"))
        return out

    # ------------------------------------------------------------------
    @property
    def critical(self) -> M.MarginResult:
        """The governing (lowest) margin."""
        return min(self.results, key=lambda r: r.MS)

    @property
    def passes(self) -> bool:
        return all(r.passes for r in self.results)

    def to_dataframe(self):
        """Results as a pandas DataFrame (pandas imported lazily)."""
        import pandas as pd

        df = pd.DataFrame([{
            "mode": r.mode,
            "basis": r.basis,
            "applied": r.applied,
            "allowable": r.allowable,
            "SF": r.SF,
            "MS": r.MS,
            "pass": r.passes,
            "note": r.note,
        } for r in self.results])
        return df.sort_values("MS").reset_index(drop=True)

    def report(self) -> str:
        lines = [
            f"Fastener            : {self.fastener.designation} "
            f"({self.fastener.head_style}), At = {self.fastener.At:.5f} in^2",
            f"Bolt material       : {self.bolt_material.name}",
            f"Configuration       : {self.joint.configuration}",
            f"Grip length L       : {self.joint.L:.4f} in",
            f"kb / kj             : {self.joint.kb:,.0f} / {self.joint.kj:,.0f} lbf/in",
            f"phi, n              : {self.joint.phi:.4f}, {self.joint.n:.4f}",
            f"Preload Po          : {self.preload.Po:.2f} lbf "
            f"(max {self.preload.Po_max:.2f}, min {self.preload.Po_min:.2f})",
            f"Nominal torque      : {self.preload.torque_nominal:.3f} lbf-in",
            f"Thermal load        : {self.P_thermal:+.2f} lbf "
            f"(dT = {self.load_case.delta_T:+.1f} F)",
            f"Bolt axial load P_b : {self.P_b:.2f} lbf",
            "",
            f"{'mode':<32}{'basis':<8}{'MS':>10}  status",
            "-" * 62,
        ]
        for r in sorted(self.results, key=lambda x: x.MS):
            ms = "inf" if r.MS == float("inf") else f"{r.MS:.3f}"
            lines.append(f"{r.mode:<32}{r.basis:<8}{ms:>10}  "
                         f"{'PASS' if r.passes else 'FAIL'}")
        lines += ["-" * 62,
                  f"Critical: {self.critical.mode} ({self.critical.basis}), "
                  f"MS = {self.critical.MS:.3f}"]
        return "\n".join(lines)
