# MechanicalEngModule

Preloaded bolted joint analysis per NASA TM-106943 (Chambers) and the
*Guideline for Bolted Joint Design and Analysis* v1.0, with random and sine
vibration loads for the no-FEM case.

Replaces the bolted-joint sections of `MachanicalEngineeringNotebookRevB.ipynb`.

## One source, two front ends

All physics, and the step that builds a model from the inputs, live in the
package. The notebook and the GUI only collect inputs and show results, so a
fix or a new check is made once, in a `.py` file, and both pick it up.

```
MechanicalEngModule/        <- the only place calculations live
   ^                ^
Notebooks/...ipynb  app.py   <- inputs + display only
   \                /
    case.json (optional: the same case saved and loaded by both)
```

## Setup (once per computer)

```bash
git clone https://github.com/vmarrero816/MechanicalEng-Module.git
cd MechanicalEng-Module
pip install -e ".[gui,notebook,test]"
```

`-e` (editable) installs a link to this folder rather than a copy, so the
notebook and the GUI always run the code in your clone. There is nothing to
reinstall after editing a file or pulling changes.

## Daily use

| Task | Command |
| --- | --- |
| Get the latest code | `git pull` (in the repo folder) |
| Notebook | open `MechanicalEngModule/Notebooks/BoltedJointAnalysis_RevC.ipynb` |
| GUI | `streamlit run MechanicalEngModule/app.py` (in a terminal, not VS Code's Run button) |
| Tests | `python -m pytest MechanicalEngModule -q` |

After editing a package file, restart the notebook kernel; Streamlit reloads
on its own.

To move a case between the two:

```python
case.to_json("my_case.json")                 # notebook: save
case = JointCase.from_json("my_case.json")   # notebook: load
```

In the GUI, use **Save case** / **Load case** at the top of the sidebar.

## Use from Python

```python
from MechanicalEngModule import JointCase

case = JointCase(thread="10-32", configuration=1, P_et=150.0, V=60.0)
model = case.build()                 # fastener, joint, preload, load case, ...
envelope = model.envelope()          # ambient / hot / cold
print(envelope.report())
envelope.envelope()                  # worst margin per mode, driving case named
model.sweep_load("P_et", range(0, 400, 20))
```

The lower-level objects (`Fastener`, `Joint`, `Preload`, `JointAnalysis`,
`run_thermal_envelope`, ...) remain available for anything `JointCase` does not
cover.

## Layout

```
MechanicalEngModule/
  fasteners.py   unified thread table, tensile stress area, thread shear areas
  materials.py   allowables, Fsu/Fsy derivation, material library
  joint.py       grip length, kb, kj, loading plane factor n
  thermal.py     Shigley frustum stack, layer-by-layer thermal load, environment
  preload.py     nut factors, uncertainty, Po_max / Po_min / relaxation
  margins.py     every failure mode, both strength bases
  analysis.py    JointAnalysis, LoadCase, hot/cold envelope
  case.py        JointCase (saveable inputs) -> JointModel (build, runs, sweeps)
  vibe.py        random/sine vibe -> interface loads -> per-bolt loads, slip, fatigue
  app.py         Streamlit GUI
  Notebooks/     BoltedJointAnalysis_RevC.ipynb, RevB (superseded), figures
  test_*.py      regression tests
```

Units: in, lbf, lbf-in, psi. Temperatures are entered in degC; temperature
changes are carried internally in degF.

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
shear tear-out, bolt bearing, and head/washer bearing. `vibe.py` adds joint
slip (NASA-STD-5020 A.10-6) and random-vibration fatigue.

## Thermal

Member stiffness can be built layer by layer from Shigley stress cones
(`stiffness_method="frustum"`, Guideline sec. 5.2) instead of one closed-form
expression with an equivalent modulus. The thermal load follows Guideline
sec. 7:

    P_th = dT * (sum(alpha_i * l_i) - alpha_b * L) / (1/k_b + 1/k_j)

Both operational extremes are run, because one temperature cannot drive both
checks: a tensile thermal load adds to maximum preload for bolt strength, and a
preload-relieving load subtracts from minimum preload for separation.

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
| 15 | Single equivalent-modulus member stiffness | Shigley frustum stack, layer by layer (optional) |
| 16 | One temperature case | hot / cold / ambient envelope, worst margin per mode |

## Before releasing an analysis

Material properties in `materials.py` are typical values for convenience.
Replace them with MMPDS / the applicable material and fastener procurement
specifications. Bearing allowables default to `1.5*Ftu`, a placeholder.

TM-106943 cautions that the combined-load margins can be unconservative for
shear-critical joints. Check the interaction curve directly when margins fall
below about 0.5, and use Bruhn or NASA TM X-73305 for tear-out and bearing when
`e/D < 1.5`.
