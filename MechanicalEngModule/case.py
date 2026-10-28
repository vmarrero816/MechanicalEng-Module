"""One analysis case, shared by the notebook and the GUI.

Units: in, lbf, lbf-in, psi; temperatures in degC.

``JointCase`` holds every input as plain numbers and strings, so a case can be
saved to JSON in one front end and loaded in the other.  ``JointCase.build()``
turns it into a ``JointModel`` -- the fastener, joint, preload, load case,
insert and abutment objects -- and the model runs the analyses and sweeps.

Both front ends call these functions; neither builds the model itself.  A
change made here shows up in the notebook and the GUI at once.

    case = JointCase(thread="10-32", P_et=150.0)       # or JointCase.from_json(path)
    model = case.build()
    envelope = model.envelope()
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field, fields

from .analysis import (AbutmentSpec, InsertSpec, JointAnalysis, LoadCase,
                       ThermalEnvelope, run_thermal_envelope)
from .fasteners import Fastener, min_engagement_equal_strength
from .joint import Joint, Layer
from .materials import Material, get as get_material
from .preload import (NUT_FACTOR, UNCERTAINTY, Preload, preload_from_torque,
                      preload_from_yield_fraction, preload_target)
from .thermal import ThermalEnvironment

__all__ = ["JointCase", "JointModel", "PRELOAD_METHODS"]

PRELOAD_METHODS = ("target", "torque", "yield_fraction")


@dataclass
class JointCase:
    """All inputs for one bolted-joint analysis.  Defaults are the Rev C example."""

    name: str = "case 1"

    # fastener
    thread: str = "2-56"
    head_style: str = "socket"                 # 'socket' | 'flat100'
    shank_dia: float | None = None             # None = full thread
    bolt_material: str = "A286"
    shear_plane_in_shank: bool = False

    # joint stack, head end first; last layer is the tapped parent for config 3/4
    configuration: int = 3
    layers: list = field(default_factory=lambda: [[0.060, "AL6061-T6"],
                                                  [0.129, "AL6061-T6"]])
    engagement_dia: float = 1.5                # thread engagement, in diameters
    stiffness_method: str = "frustum"          # 'frustum' | 'tm106943'
    washer_face_dia: float | None = None       # None = head diameter
    cone_angle_deg: float = 30.0

    # preload
    preload_method: str = "target"             # 'target' | 'torque' | 'yield_fraction'
    target_preload: float = 81.0               # lbf
    applied_torque: float = 2.1                # lbf-in
    yield_fraction: float = 0.65
    install: str = "dry"                       # NUT_FACTOR key
    uncertainty_key: str = "torque wrench, unlubricated"
    relaxation: float = 0.05

    # load case, per fastener
    P_et: float = 4.0                          # lbf
    V: float = 2.0                             # lbf
    Mb: float = 0.0                            # lbf-in
    T_assembly_C: float = 20.0
    T_hot_C: float = 71.0
    T_cold_C: float = -25.0
    SF_tension: float = 1.4
    SF_shear: float = 1.4
    SF_separation: float = 1.2                 # 1.2 structural, 1.4 pressure

    # threaded insert -- used for configurations 3 and 4 only
    insert_material: str = "CRES304"
    insert_major_dia: float = 0.1380           # insert EXTERNAL thread major dia
    le_insert_parent: float = 0.129            # insert-into-parent engagement

    # abutment (tear-out and bearing)
    use_abutment: bool = True
    abutment_material: str = "AL6061-T6"
    sheet_thickness: float = 0.060
    edge_distance: float = 0.200
    hole_dia: float = 0.096
    washer_od: float | None = 0.096            # None = no washer

    number_of_bolts: int = 7                   # reporting only

    # -- save / load ---------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = [[float(t), str(m)] for t, m in self.layers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JointCase":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown case fields: {unknown}")
        d = dict(d)
        if "layers" in d:
            d["layers"] = [[float(t), str(m)] for t, m in d["layers"]]
        return cls(**d)

    def to_json(self, path: str | pathlib.Path | None = None) -> str:
        """JSON text of the case; also written to ``path`` when given."""
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            pathlib.Path(path).write_text(text + "\n")
        return text

    @classmethod
    def from_json(cls, source: str | pathlib.Path) -> "JointCase":
        """Load from a JSON file path or a JSON string."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = pathlib.Path(source).read_text()
        return cls.from_dict(json.loads(text))

    # -- build ---------------------------------------------------------------
    def build(self) -> "JointModel":
        """Construct every analysis object from the inputs."""
        if self.preload_method not in PRELOAD_METHODS:
            raise ValueError(f"preload_method must be one of {PRELOAD_METHODS}")
        if self.configuration not in (1, 2, 3, 4):
            raise ValueError("configuration must be 1, 2, 3 or 4")
        if self.install not in NUT_FACTOR:
            raise ValueError(f"install {self.install!r} has no GSFC 540-PG nut factor; "
                             f"use one of {sorted(NUT_FACTOR)}")

        fastener = Fastener(self.thread, head_style=self.head_style,
                            shank_dia=self.shank_dia)
        bolt_mat = get_material(self.bolt_material)
        layers = [Layer(float(t), get_material(m)) for t, m in self.layers]
        tapped = self.configuration in (3, 4)
        Li = self.engagement_dia * fastener.d_major if tapped else None
        joint = Joint(fastener, layers, bolt_mat, configuration=self.configuration,
                      insert_length=Li, stiffness_method=self.stiffness_method,
                      washer_face_dia=self.washer_face_dia,
                      cone_angle_deg=self.cone_angle_deg)
        environment = ThermalEnvironment(T_assembly=self.T_assembly_C,
                                         T_hot=self.T_hot_C, T_cold=self.T_cold_C)

        K, u, r = NUT_FACTOR[self.install], UNCERTAINTY[self.uncertainty_key], self.relaxation
        if self.preload_method == "target":
            preload = preload_target(self.target_preload, fastener, K=K, u=u, relax=r)
        elif self.preload_method == "torque":
            preload = preload_from_torque(self.applied_torque, K, fastener.d_major,
                                          u=u, relax=r)
        else:
            preload = preload_from_yield_fraction(fastener, bolt_mat, self.yield_fraction,
                                                  K=K, u=u, relax=r)

        load_case = LoadCase(self.name, P_et=self.P_et, V=self.V, Mb=self.Mb,
                             delta_T=environment.cases["hot"],
                             SF_tension=self.SF_tension, SF_shear=self.SF_shear,
                             SF_separation=self.SF_separation)
        insert = (InsertSpec(get_material(self.insert_material), layers[-1].material,
                             self.insert_major_dia, Li, self.le_insert_parent)
                  if tapped else None)
        abutment = (AbutmentSpec(get_material(self.abutment_material),
                                 self.sheet_thickness, self.edge_distance,
                                 self.hole_dia, self.washer_od or None)
                    if self.use_abutment else None)
        return JointModel(self, fastener, bolt_mat, layers, Li, joint, preload,
                          environment, load_case, insert, abutment)


@dataclass
class JointModel:
    """The built analysis objects for one case, plus the runs both front ends use."""

    case: JointCase
    fastener: Fastener
    bolt_mat: Material
    layers: list[Layer]
    Li: float | None
    joint: Joint
    preload: Preload
    environment: ThermalEnvironment
    load_case: LoadCase
    insert: InsertSpec | None
    abutment: AbutmentSpec | None

    def analysis(self, temperature: str = "hot", preload: Preload | None = None,
                 **load_overrides) -> JointAnalysis:
        """Single-point margin set at one temperature case ('ambient'|'hot'|'cold').

        ``load_overrides`` replace LoadCase fields, e.g. ``P_et=40.0``.
        """
        lc = LoadCase(**{**self.load_case.__dict__,
                         "delta_T": self.environment.cases[temperature],
                         **load_overrides})
        return JointAnalysis(self.fastener, self.bolt_mat, self.joint,
                             preload or self.preload, lc, insert=self.insert,
                             abutment=self.abutment,
                             shear_plane_in_shank=self.case.shear_plane_in_shank)

    def envelope(self) -> ThermalEnvelope:
        """Full margin set at ambient, hot and cold."""
        return run_thermal_envelope(self.fastener, self.bolt_mat, self.joint,
                                    self.preload, self.load_case, self.environment,
                                    insert=self.insert, abutment=self.abutment,
                                    shear_plane_in_shank=self.case.shear_plane_in_shank)

    def sweep_load(self, attr: str, values, temperature: str = "hot"):
        """Governing margin as one load is swept ('P_et', 'V' or 'Mb')."""
        import pandas as pd

        if attr not in ("P_et", "V", "Mb"):
            raise ValueError("attr must be 'P_et', 'V' or 'Mb'")
        rows = []
        for v in values:
            a = self.analysis(temperature, **{attr: float(v)})
            rows.append({attr: float(v), "critical MS": a.critical.MS,
                         "mode": a.critical.mode})
        return pd.DataFrame(rows)

    def sweep_preload(self, values, temperature: str = "hot"):
        """Governing, bolt-tension and separation margins vs nominal preload."""
        import pandas as pd

        p0 = self.preload
        rows = []
        for Po in values:
            p = Preload(Po=float(Po), u=p0.u, relax=p0.relax, K=p0.K, d_major=p0.d_major)
            a = self.analysis(temperature, preload=p)
            by = {(r.mode, r.basis): r.MS for r in a.results}
            rows.append({"Po": float(Po), "critical": a.critical.MS,
                         "bolt tension (yield)": by[("bolt tension", "yield")],
                         "separation": by[("joint separation", "n/a")]})
        return pd.DataFrame(rows)

    def warnings(self, envelope: ThermalEnvelope | None = None) -> list[str]:
        """Design checks worth flagging alongside the margins."""
        out = []
        Le_min = min_engagement_equal_strength(self.fastener)
        if self.Li and self.Li < Le_min:
            out.append(f"Engagement {self.Li:.4f} in is below the equal-strength "
                       f"minimum {Le_min:.4f} in.")
        if self.abutment:
            eD = self.abutment.edge_distance / self.fastener.d_major
            if eD < 1.5:
                out.append(f"e/D = {eD:.2f}, below 1.5. Use Bruhn / TM X-73305.")
        if envelope is not None and abs(envelope.critical.MS) < 0.5:
            out.append("Governing margin below 0.5. Check the interaction curve "
                       "directly; the approximation can be unconservative.")
        return out
