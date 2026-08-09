# boltjoint

Preloaded bolted joint analysis per NASA TM-106943 (Chambers) and the
*Guideline for Bolted Joint Design and Analysis* v1.0.

Replaces the bolted-joint sections of `MachanicalEngineeringNotebookRevB.ipynb`.
All physics lives in the package; the notebook is a thin front end, so the same
code can drive a GUI later without duplication.

## Layout

```
boltjoint/
  fasteners.py   unified thread table, tensile stress area, thread shear areas
  materials.py   allowables, Fsu/Fsy derivation, material library
  joint.py       grip length, kb, kj, loading plane factor n, thermal load
  preload.py     nut factors, uncertainty, Po_max / Po_min / relaxation
  margins.py     every failure mode, both strength bases
  analysis.py    orchestrator -> report + DataFrame
tests/           26 regression tests, each pinning a Rev B defect
notebooks/       BoltedJointAnalysis_RevC.ipynb
app.py           Streamlit front end
```

Two front ends, one set of physics. The notebook is for exploratory and
documented work; the app is for quick trades and design reviews.

```bash
pip install -e ".[gui]"
streamlit run app.py
```

Units are inches, lbf, psi, degF throughout.

## Use

```python
from boltjoint import *

f     = Fastener("2-56")
bolt  = materials.get("A286")
al    = materials.get("AL6061-T6")

joint = Joint(f, [Layer(0.060, al), Layer(0.129, al)], bolt,
              configuration=3, insert_length=0.129)
p     = preload_target(81.0, f, K=0.20, u=0.35, relax=0.05)

a = JointAnalysis(f, bolt, joint, p,
                  LoadCase(P_et=4.0, V=2.0, delta_T=-81.0, SF_tension=1.4))
print(a.report())
a.to_dataframe()
```

## Design rules enforced

1. The factor of safety multiplies **external loads only**. Preload and thermal
   loads are fixed quantities and are never scaled (Guideline sec. 11).
2. Bolt strength checks use **maximum** preload; separation uses **minimum**
   preload less relaxation.
3. Tension and shear allowables use the **tensile stress area** `At`, never the
   thread shear area.
4. Yield and ultimate bases are evaluated **separately** and never mixed inside
   one interaction equation.
5. No module-level state. Every function takes explicit arguments, so no
   variable can be silently redefined by an out-of-order cell.

## Failure modes covered

Bolt tension, bolt shear, combined tension/shear/bending, joint separation,
bolt external thread shear, the three threaded-insert pull-out modes, sheet
shear tear-out, bolt bearing, and head/washer bearing.

## Corrections relative to Rev B

| # | Rev B | Rev C |
|---|-------|-------|
| 1 | Tension and shear allowables used the thread shear area `5*pi*Le*D_min/8` (4.6x too large for a #2-56) | tensile stress area `At` |
| 2 | `MS = np.max(P_ult)/P_b - 1` for insert pull-out | `min` of the three modes |
| 3 | `Stress = P_b * A` | `P_b / A` |
| 4 | `P_b = P_min + P_external` | `P_b = Po_max + P_thermal + n*phi*SF*P_et` |
| 5 | `TensionSafetyFactor` defined but never applied | SF applied to external loads everywhere |
| 6 | `n` was both the loading-plane factor and threads per inch | no shared globals |
| 7 | `Fty` in `Rt` but `Fsu` in `Rs` | bases evaluated separately |
| 8 | `Fsu = 0.75*Ftu` | 0.55-0.60 `Ftu`; `Fsy = 0.577*Fty` |
| 9 | `np.round(At, 3)` | no rounding |
| 10 | `P_relax` commented out; uncertainty was 0.35, 0.05 and 0 in three cells | one `Preload` object |
| 11 | Thermal load back-solved into a fictitious external load | separate signed term |
| 12 | Separation SF = 1.25 with the wrong `P_sep` | 1.2 / 1.4 with `P_sep = (1-n*phi)*P_et` |
| 13 | Insert modes reused the bolt minor diameter and `Le` | correct mating geometry per mode |
| 14 | Configurations 2 and 4 raised `IndexError` / `KeyError` | fixed and guarded |

## Verification

```bash
python -m pytest tests -q
```

## Before releasing an analysis

Material properties in `materials.py` are typical values for convenience.
Replace them with MMPDS / the applicable material and fastener procurement
specifications. Bearing allowables default to `1.5*Ftu`, a placeholder.

TM-106943 cautions that the combined-load margins can be unconservative for
shear-critical joints. Check the interaction curve directly when margins fall
below about 0.5, and use Bruhn or NASA TM X-73305 for tear-out and bearing when
`e/D < 1.5`.
