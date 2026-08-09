"""boltjoint -- preloaded bolted joint analysis.

Implements NASA TM-106943 (Chambers) and the Guideline for Bolted Joint Design
and Analysis v1.0.  All quantities in inches, lbf and psi unless noted.

Design rules enforced throughout:
  * The factor of safety multiplies EXTERNAL loads only, never preload.
  * Bolt strength checks use MAXIMUM preload; separation uses MINIMUM preload.
  * Tension and shear allowables use the tensile stress area At, not the
    thread shear area.
  * Yield and ultimate bases are always evaluated separately.
  * No global state -- every function takes explicit arguments.

Quick start
-----------
>>> from boltjoint import *
>>> f = Fastener("2-56")
>>> bolt = materials.get("A286")
>>> al = materials.get("AL6061-T6")
>>> j = Joint(f, [Layer(0.060, al), Layer(0.129, al)], bolt,
...           configuration=3, insert_length=0.129)
>>> p = preload_from_yield_fraction(f, bolt, 0.65, K=0.20, u=0.35)
>>> a = JointAnalysis(f, bolt, j, p, LoadCase(P_et=4.0, V=2.0, delta_T=-81.0))
>>> print(a.report())
"""

from . import fasteners, joint, margins, materials, preload
from .analysis import AbutmentSpec, InsertSpec, JointAnalysis, LoadCase
from .fasteners import (
    Fastener,
    bolt_thread_shear_area,
    internal_thread_shear_area,
    min_engagement_equal_strength,
)
from .joint import Joint, Layer
from .margins import BASES, MarginResult, bolt_axial_load, load_ratios
from .materials import Material
from .preload import (
    NUT_FACTOR,
    UNCERTAINTY,
    Preload,
    preload_from_torque,
    preload_from_yield_fraction,
    preload_target,
)

__version__ = "1.0.0"

__all__ = [
    "Fastener", "Material", "Layer", "Joint", "Preload",
    "LoadCase", "InsertSpec", "AbutmentSpec", "JointAnalysis",
    "MarginResult", "bolt_axial_load", "load_ratios",
    "preload_from_torque", "preload_from_yield_fraction", "preload_target",
    "bolt_thread_shear_area", "internal_thread_shear_area",
    "min_engagement_equal_strength",
    "NUT_FACTOR", "UNCERTAINTY", "BASES",
    "fasteners", "materials", "joint", "preload", "margins",
]
