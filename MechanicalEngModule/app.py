"""Bolted joint analysis - interactive front end.

    streamlit run MechanicalEngModule/app.py        (from the repo root)

All physics and the model build come from MechanicalEngModule (see case.py);
this module only collects inputs and renders results.  No calculation lives
here, so the notebook and this app always agree.  A case saved here as JSON
loads in the notebook with ``JointCase.from_json``, and the reverse.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import MechanicalEngModule  # noqa: F401  (installed with `pip install -e .`)
except ModuleNotFoundError:     # not installed: use the repo copy this file sits in
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from MechanicalEngModule import (  # noqa: E402
    JointCase, NUT_FACTOR, UNCERTAINTY, materials, margins as M,
)
from MechanicalEngModule.fasteners import THREADS  # noqa: E402

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#14202E"
PAPER = "#EEF1F3"
CARD = "#FFFFFF"
RULE = "#C3CCD3"
BLUE = "#2E5FA3"
PASS = "#1F6F5C"
FAIL = "#B3261E"
CAUTION = "#C77D0A"
MUTED = "#5C6B7A"

st.set_page_config(page_title="Bolted Joint Analysis",
                   page_icon="⬡", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}

/* ---- title block: mirrors a drawing title block ---- */
.bj-titleblock {{
  display: grid; grid-template-columns: 1fr auto;
  border: 1.5px solid {INK}; background: {CARD}; margin-bottom: 1.6rem;
}}
.bj-tb-main {{ padding: 1.1rem 1.4rem; }}
.bj-tb-eyebrow {{
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .18em;
  text-transform: uppercase; color: {MUTED}; margin-bottom: .35rem;
}}
.bj-tb-title {{
  font-family: 'Archivo Narrow', sans-serif; font-weight: 700; font-size: 2.1rem;
  line-height: 1.05; letter-spacing: -.01em; margin: 0;
}}
.bj-tb-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: {MUTED}; margin-top: .5rem; }}
.bj-tb-fields {{ display: flex; border-left: 1.5px solid {INK}; }}
.bj-tb-field {{ padding: .9rem 1.1rem; border-right: 1px solid {RULE}; min-width: 108px; }}
.bj-tb-field:last-child {{ border-right: none; }}
.bj-tb-key {{
  font-family: 'IBM Plex Mono', monospace; font-size: .6rem; letter-spacing: .14em;
  text-transform: uppercase; color: {MUTED};
}}
.bj-tb-val {{
  font-family: 'IBM Plex Mono', monospace; font-size: 1.02rem; font-weight: 600;
  font-variant-numeric: tabular-nums; margin-top: .2rem;
}}

/* ---- inspection stamp: the signature element ---- */
.bj-stamp {{
  display: inline-block; transform: rotate(-6deg);
  border: 3px double var(--c); color: var(--c);
  padding: .5rem 1.3rem; border-radius: 3px; text-align: center;
  font-family: 'Archivo Narrow', sans-serif; font-weight: 700;
  font-size: 1.25rem; letter-spacing: .18em; text-transform: uppercase;
  line-height: 1.25; opacity: .92;
}}
.bj-stamp small {{ display: block; font-size: .62rem; letter-spacing: .2em; font-weight: 500; }}

/* ---- spec cards ---- */
.bj-cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0;
  border: 1px solid {RULE}; background: {CARD}; }}
.bj-card {{ padding: .85rem 1rem; border-right: 1px solid {RULE}; }}
.bj-card:last-child {{ border-right: none; }}
.bj-card-k {{ font-family: 'IBM Plex Mono', monospace; font-size: .6rem;
  letter-spacing: .13em; text-transform: uppercase; color: {MUTED}; }}
.bj-card-v {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.28rem;
  font-weight: 600; font-variant-numeric: tabular-nums; margin: .18rem 0; }}
.bj-card-s {{ font-family: 'IBM Plex Mono', monospace; font-size: .66rem; color: {MUTED}; }}
.bj-stamp-note {{
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: {MUTED};
  letter-spacing: .1em; margin-top: .9rem; text-transform: uppercase;
}}

/* ---- margin ladder ---- */
.bj-ladder {{ background: {CARD}; border: 1px solid {RULE}; padding: 1.2rem 1.4rem; }}
.bj-rung {{ display: grid; grid-template-columns: 210px 1fr 74px; align-items: center;
  gap: .8rem; padding: .26rem 0; }}
.bj-rung-label {{
  font-family: 'IBM Plex Mono', monospace; font-size: .74rem; color: {INK};
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.bj-rung-label small {{ color: {MUTED}; }}
.bj-track {{ position: relative; height: 15px; background: {PAPER}; border: 1px solid {RULE}; }}
.bj-zero {{ position: absolute; left: 14%; top: -3px; bottom: -3px; width: 2px; background: {FAIL}; z-index: 3; }}
.bj-tick {{ position: absolute; top: 0; bottom: 0; width: 1px; background: {RULE}; }}
.bj-negzone {{ position: absolute; left: 0; top: 0; bottom: 0; width: 14%;
  background: repeating-linear-gradient(-45deg, #FBE9E7, #FBE9E7 3px, {PAPER} 3px, {PAPER} 6px); }}
.bj-bar {{ position: absolute; top: 0; bottom: 0; background: var(--c); opacity: .85; }}
.bj-scale {{ position: relative; height: 15px; margin-top: .35rem; }}
.bj-scale span {{
  position: absolute; transform: translateX(-50%); font-family: 'IBM Plex Mono', monospace;
  font-size: .6rem; color: {MUTED};
}}
.bj-rung-ms {{
  font-family: 'IBM Plex Mono', monospace; font-size: .8rem; font-weight: 600;
  font-variant-numeric: tabular-nums; text-align: right; color: var(--c);
}}
.bj-gov {{ background: #FFF8E8; box-shadow: inset 3px 0 0 {CAUTION}; padding-left: .5rem;
  margin-left: -.5rem; }}

/* ---- section headings ---- */
.bj-h {{
  font-family: 'Archivo Narrow', sans-serif; font-weight: 600; font-size: 1.15rem;
  letter-spacing: .02em; margin: 2rem 0 .7rem 0;
  padding-bottom: .3rem; border-bottom: 1.5px solid {INK};
}}
.bj-note {{ font-family: 'IBM Plex Sans'; font-size: .8rem; color: {MUTED}; margin: .5rem 0 1rem 0; }}
.bj-warn {{
  font-family: 'IBM Plex Mono', monospace; font-size: .76rem; color: {CAUTION};
  border-left: 3px solid {CAUTION}; padding: .5rem .8rem; background: #FFF8E8; margin: .6rem 0;
}}
.bj-err {{
  font-family: 'IBM Plex Mono', monospace; font-size: .8rem; color: {FAIL};
  border-left: 3px solid {FAIL}; padding: .8rem 1rem; background: #FDF0EF;
}}
section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1.5px solid {INK}; }}
section[data-testid="stSidebar"] h2 {{
  font-family: 'Archivo Narrow', sans-serif; font-size: .82rem; letter-spacing: .14em;
  text-transform: uppercase; color: {MUTED}; margin-top: 1.1rem;
}}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; }}
</style>
""", unsafe_allow_html=True)

MAT_KEYS = sorted(materials.LIBRARY)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
sb = st.sidebar


def _index(options, value, fallback=0):
    """Position of ``value`` in ``options`` (for widget defaults)."""
    options = list(options)
    return options.index(value) if value in options else fallback


# -- case file: a loaded case becomes the default for every widget --------
sb.markdown("## Case file")
upload = sb.file_uploader("Load case (.json)", type="json",
                          help="A case saved from this app or from the notebook "
                               "with JointCase.to_json().")
base, token = JointCase(), "default"
if upload is not None:
    raw = upload.getvalue()
    try:
        base = JointCase.from_json(raw.decode())
        token = hashlib.md5(raw).hexdigest()[:10]
    except (ValueError, KeyError, TypeError) as exc:
        sb.error(f"Cannot read that case file: {exc}")


def key(name: str) -> str:
    """Widget key; changes when a new case file is loaded so widgets reset."""
    return f"{name}-{token}"


case_name = sb.text_input("Case name", base.name, key=key("name"))

sb.markdown("## Fastener")
THREAD_KEYS = sorted(THREADS)
thread = sb.selectbox("Thread", THREAD_KEYS, index=_index(THREAD_KEYS, base.thread),
                      key=key("thread"))
head_style = sb.selectbox("Head style", ["socket", "flat100"],
                          index=_index(["socket", "flat100"], base.head_style),
                          key=key("head"))
bolt_key = sb.selectbox("Bolt material", MAT_KEYS,
                        index=_index(MAT_KEYS, base.bolt_material), key=key("boltmat"))
reduced = sb.checkbox("Reduced shank", value=base.shank_dia is not None, key=key("reduced"))
shank_dia = sb.number_input("Shank diameter [in]", 0.01, 1.0,
                            float(base.shank_dia or THREADS[thread][0]), format="%.4f",
                            key=key("shank")) if reduced else None
shank_shear = sb.checkbox("Shear plane on shank", value=base.shear_plane_in_shank,
                          help="Otherwise the threads are in the shear plane and At is used.",
                          key=key("shankshear"))

sb.markdown("## Joint stack")
config = sb.selectbox("Configuration", [1, 2, 3, 4],
                      index=_index([1, 2, 3, 4], base.configuration, 2),
                      format_func=lambda c: {
                          1: "1 - through-bolt + nut",
                          2: "2 - through-bolt + nut, flat head",
                          3: "3 - insert / tapped hole",
                          4: "4 - insert / tapped hole, flat head"}[c],
                      key=key("config"))
n_layers = sb.number_input("Number of layers", 2, 5, min(max(len(base.layers), 2), 5),
                           key=key("nlayers"))
layers_in = []
for i in range(int(n_layers)):
    last = i == n_layers - 1
    tag = "parent" if (last and config in (3, 4)) else f"layer {i + 1}"
    t0, m0 = base.layers[i] if i < len(base.layers) else (0.129 if last else 0.060,
                                                          "AL6061-T6")
    c1, c2 = sb.columns([1, 1.3])
    t = c1.number_input(f"t {tag} [in]", 0.001, 5.0, float(t0), format="%.4f",
                        key=key(f"t{i}"))
    m = c2.selectbox(f"mat {tag}", MAT_KEYS, index=_index(MAT_KEYS, m0), key=key(f"m{i}"))
    layers_in.append([t, m])
engage_dia = sb.number_input("Engagement [diameters]", 0.5, 4.0, float(base.engagement_dia),
                             0.1, key=key("engage")) if config in (3, 4) else None
stiff_method = sb.radio(
    "Member stiffness", ["frustum", "tm106943"],
    index=_index(["frustum", "tm106943"], base.stiffness_method),
    format_func=lambda m: {"frustum": "Shigley frustum (layer by layer)",
                           "tm106943": "TM-106943 equivalent modulus"}[m],
    help="The frustum method builds the stack cone by cone and is preferred "
         "for dissimilar materials.", key=key("stiff"))
cone_angle = sb.slider("Cone half-angle [deg]", 25.0, 33.0, float(base.cone_angle_deg), 0.5,
                       key=key("cone")) if stiff_method == "frustum" else 30.0

sb.markdown("## Preload")
METHOD_LABELS = {"target": "target force", "torque": "applied torque",
                 "yield_fraction": "fraction of yield"}
method = sb.radio("Basis", list(METHOD_LABELS),
                  index=_index(list(METHOD_LABELS), base.preload_method),
                  format_func=METHOD_LABELS.get, key=key("method"))
target_P = sb.number_input("Target preload [lbf]", 0.0, 1e5, float(base.target_preload),
                           key=key("targetP")) if method == "target" else base.target_preload
applied_T = sb.number_input("Applied torque [lbf-in]", 0.0, 1e5, float(base.applied_torque),
                            key=key("torque")) if method == "torque" else base.applied_torque
yield_frac = sb.slider("Fraction of bolt yield", 0.2, 0.9, float(base.yield_fraction), 0.01,
                       key=key("yfrac")) if method == "yield_fraction" else base.yield_fraction
install = sb.selectbox("Installation", sorted(NUT_FACTOR),
                       index=_index(sorted(NUT_FACTOR), base.install), key=key("install"))
unc_key = sb.selectbox("Preload uncertainty", sorted(UNCERTAINTY),
                       index=_index(sorted(UNCERTAINTY), base.uncertainty_key),
                       key=key("unc"))
relax = sb.slider("Short-term relaxation", 0.0, 0.15, float(base.relaxation), 0.01,
                  key=key("relax"))

sb.markdown("## Applied loads")
P_et = sb.number_input("Axial load P_et [lbf]", 0.0, 1e6, float(base.P_et), key=key("Pet"))
V = sb.number_input("Shear load V [lbf]", 0.0, 1e6, float(base.V), key=key("V"))
Mb = sb.number_input("Bending moment [lbf-in]", 0.0, 1e6, float(base.Mb), key=key("Mb"))
T_asm = sb.number_input("Assembly temperature [C]", -273.0, 500.0, float(base.T_assembly_C),
                        key=key("Tasm"))
T_hot = sb.number_input("Hot limit [C]", -273.0, 800.0, float(base.T_hot_C), key=key("Thot"))
T_cold = sb.number_input("Cold limit [C]", -273.0, 500.0, float(base.T_cold_C),
                         key=key("Tcold"))
sf_t = sb.number_input("SF tension", 1.0, 5.0, float(base.SF_tension), 0.05, key=key("sft"))
sf_s = sb.number_input("SF shear", 1.0, 5.0, float(base.SF_shear), 0.05, key=key("sfs"))
sf_sep = sb.number_input("SF separation", 1.0, 3.0, float(base.SF_separation), 0.05,
                         help="1.2 structural, 1.4 pressure systems", key=key("sfsep"))

ins_key, ins_major, le_ins_par = (base.insert_material, base.insert_major_dia,
                                  base.le_insert_parent)
if config in (3, 4):
    sb.markdown("## Threaded insert")
    ins_key = sb.selectbox("Insert material", MAT_KEYS,
                           index=_index(MAT_KEYS, base.insert_material), key=key("insmat"))
    ins_major = sb.number_input("Insert external major dia [in]", 0.01, 2.0,
                                float(base.insert_major_dia), format="%.4f",
                                key=key("insmaj"))
    le_ins_par = sb.number_input("Insert-into-parent engagement [in]", 0.01, 2.0,
                                 float(base.le_insert_parent), format="%.4f",
                                 key=key("insle"))

sb.markdown("## Abutment")
use_ab = sb.checkbox("Check tear-out and bearing", value=base.use_abutment, key=key("useab"))
ab_key, sheet_t, edge_d, hole_d, washer_od = (base.abutment_material, base.sheet_thickness,
                                              base.edge_distance, base.hole_dia,
                                              base.washer_od)
if use_ab:
    ab_key = sb.selectbox("Abutment material", MAT_KEYS,
                          index=_index(MAT_KEYS, base.abutment_material), key=key("abmat"))
    sheet_t = sb.number_input("Sheet thickness [in]", 0.005, 2.0, float(base.sheet_thickness),
                              format="%.4f", key=key("sheet"))
    edge_d = sb.number_input("Edge distance e [in]", 0.01, 5.0, float(base.edge_distance),
                             format="%.4f", key=key("edge"))
    hole_d = sb.number_input("Through-hole dia [in]", 0.01, 2.0, float(base.hole_dia),
                             format="%.4f", key=key("hole"))
    washer_od = sb.number_input("Washer OD [in], 0 = none", 0.0, 3.0,
                                float(base.washer_od or 0.0), format="%.4f",
                                key=key("washer")) or None

n_bolts = sb.number_input("Bolts in assembly (reporting)", 1, 500, int(base.number_of_bolts),
                          key=key("nbolts"))

case = JointCase(
    name=case_name, thread=thread, head_style=head_style, shank_dia=shank_dia,
    bolt_material=bolt_key, shear_plane_in_shank=shank_shear,
    configuration=config, layers=layers_in,
    engagement_dia=engage_dia if engage_dia is not None else base.engagement_dia,
    stiffness_method=stiff_method, washer_face_dia=base.washer_face_dia,
    cone_angle_deg=cone_angle,
    preload_method=method, target_preload=target_P, applied_torque=applied_T,
    yield_fraction=yield_frac, install=install, uncertainty_key=unc_key,
    relaxation=relax,
    P_et=P_et, V=V, Mb=Mb, T_assembly_C=T_asm, T_hot_C=T_hot, T_cold_C=T_cold,
    SF_tension=sf_t, SF_shear=sf_s, SF_separation=sf_sep,
    insert_material=ins_key, insert_major_dia=ins_major, le_insert_parent=le_ins_par,
    use_abutment=use_ab, abutment_material=ab_key, sheet_thickness=sheet_t,
    edge_distance=edge_d, hole_dia=hole_d, washer_od=washer_od,
    number_of_bolts=int(n_bolts),
)
sb.download_button("Save case (.json)", case.to_json(),
                   file_name=f"{(case_name or 'case').replace(' ', '_')}.json",
                   mime="application/json")

# ---------------------------------------------------------------------------
# Model -- built by the package, exactly as the notebook builds it
# ---------------------------------------------------------------------------
try:
    model = case.build()
    envelope = model.envelope()
except (ValueError, KeyError) as exc:
    st.markdown(f'<div class="bj-err"><b>Cannot build the joint.</b><br>{exc}</div>',
                unsafe_allow_html=True)
    st.stop()

fastener, bolt_mat, joint = model.fastener, model.bolt_mat, model.joint
preload, environment, load_case = model.preload, model.environment, model.load_case
analysis = envelope.cases["hot"]

crit = envelope.critical
ok = envelope.passes
crit_case = envelope.critical_case
verdict_colour = PASS if ok else FAIL

# ---------------------------------------------------------------------------
# Title block
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="bj-titleblock">
  <div class="bj-tb-main">
    <div class="bj-tb-eyebrow">Preloaded joint &middot; NASA TM-106943 &middot; Guideline v1.0</div>
    <h1 class="bj-tb-title">{thread} &nbsp;&middot;&nbsp; configuration {config}</h1>
    <div class="bj-tb-sub">{bolt_mat.name} &nbsp;|&nbsp; grip {joint.L:.4f} in
      &nbsp;|&nbsp; At {fastener.At:.5f} in&sup2; &nbsp;|&nbsp; {int(n_bolts)} places</div>
  </div>
  <div class="bj-tb-fields">
    <div class="bj-tb-field"><div class="bj-tb-key">Torque</div>
      <div class="bj-tb-val">{preload.torque_nominal:.2f}</div>
      <div class="bj-tb-key">lbf-in</div></div>
    <div class="bj-tb-field"><div class="bj-tb-key">Bolt load</div>
      <div class="bj-tb-val">{analysis.P_b:.1f}</div>
      <div class="bj-tb-key">lbf</div></div>
    <div class="bj-tb-field"><div class="bj-tb-key">Governing MS</div>
      <div class="bj-tb-val" style="color:{verdict_colour}">{crit.MS:.3f}</div>
      <div class="bj-tb-key">{crit.basis} / {crit_case}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stamp + key quantities
# ---------------------------------------------------------------------------
left, right = st.columns([1, 2.1])
with left:
    st.markdown(
        f'<div class="bj-stamp" style="--c:{verdict_colour}">'
        f'{"all margins positive" if ok else "negative margin"}'
        f'<small>{len(analysis.results)} modes checked</small></div>'
        f'<div class="bj-stamp-note">governing &mdash; {crit.mode} '
        f'({crit.basis}) in the {crit_case} case</div>',
        unsafe_allow_html=True)
    for w in model.warnings(envelope):
        st.markdown(f'<div class="bj-warn">{w}</div>', unsafe_allow_html=True)

with right:
    cards = [
        ("Preload Po", f"{preload.Po:,.1f} lbf",
         f"max {preload.Po_max:,.1f} / min {preload.Po_min:,.1f}"),
        ("Thermal load", f"{envelope.cases['hot'].P_thermal:+,.1f} lbf hot",
         f"{envelope.cases['cold'].P_thermal:+,.1f} lbf cold"),
        ("Stiffness ratio phi", f"{joint.phi:.4f}", f"loading plane n = {joint.n:.3f}"),
        ("Total clamp", f"{preload.Po * n_bolts:,.0f} lbf", f"{int(n_bolts)} fasteners"),
    ]
    st.markdown('<div class="bj-cards">' + "".join(
        f'<div class="bj-card"><div class="bj-card-k">{k}</div>'
        f'<div class="bj-card-v">{v}</div>'
        f'<div class="bj-card-s">{sub}</div></div>' for k, v, sub in cards
    ) + '</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Margin ladder
# ---------------------------------------------------------------------------
st.markdown('<div class="bj-h">Margins of safety</div>', unsafe_allow_html=True)
st.markdown('<div class="bj-note">Worst margin per mode across ambient, hot and '
            'cold, both strength bases. The driving temperature case is named '
            'beside each mode. The red line is MS = 0; the scale compresses '
            'with increasing margin so the design-driving band near zero stays '
            'readable.</div>', unsafe_allow_html=True)

# saturating scale f(ms) = ms/(1+ms): 0 -> 0, 0.5 -> .33, 1 -> .5, 3 -> .75, inf -> 1
NEG, POS = 14.0, 86.0          # percent of track width either side of zero


def _pos(ms: float) -> float:
    return POS * (ms / (1.0 + ms)) if ms != float("inf") else POS


TICKS = [(0.5, "0.5"), (1.0, "1"), (3.0, "3"), (10.0, "10")]
tick_html = "".join(
    f'<div class="bj-tick" style="left:{NEG + _pos(v):.2f}%"></div>' for v, _ in TICKS)

rungs = []
for r, r_case in envelope.worst():
    ms = r.MS
    colour = FAIL if ms < 0 else (CAUTION if ms < 0.5 else PASS)
    gov = (" bj-gov" if (r.mode, r.basis) == (crit.mode, crit.basis)
           else "")
    txt = "&infin;" if ms == float("inf") else f"{ms:,.3f}"
    if ms >= 0:
        bar = (f'<div class="bj-bar" style="--c:{colour};background:{colour};'
               f'left:{NEG:.2f}%;width:{max(_pos(ms), 0.6):.2f}%"></div>')
    else:
        w = min(abs(ms), 1.0) * NEG
        bar = (f'<div class="bj-bar" style="background:{FAIL};'
               f'left:{NEG - w:.2f}%;width:{w:.2f}%"></div>')
    rungs.append(
        f'<div class="bj-rung{gov}">'
        f'<div class="bj-rung-label">{r.mode} '
        f'<small>{r.basis} &middot; {r_case}</small></div>'
        f'<div class="bj-track"><div class="bj-negzone"></div>{tick_html}'
        f'<div class="bj-zero"></div>{bar}</div>'
        f'<div class="bj-rung-ms" style="--c:{colour};color:{colour}">{txt}</div></div>')

rungs.append('<div class="bj-rung"><div class="bj-rung-label"></div>'
             '<div class="bj-scale">'
             + f'<span style="left:{NEG:.2f}%">0</span>'
             + "".join(f'<span style="left:{NEG + _pos(v):.2f}%">{lbl}</span>'
                       for v, lbl in TICKS)
             + '</div><div class="bj-rung-ms"></div></div>')

st.markdown(f'<div class="bj-ladder">{"".join(rungs)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------
tab_tab, tab_therm, tab_int, tab_sens, tab_joint = st.tabs(
    ["Results table", "Thermal", "Interaction", "Sensitivity", "Joint properties"])

with tab_tab:
    df = envelope.envelope()
    st.dataframe(
        df.style.apply(
            lambda row: [f"background-color: #FDF0EF" if not row["pass"] else ""
                         for _ in row], axis=1
        ).format({"MS": "{:,.3f}"}),
        width="stretch", height=560)
    st.markdown('<div class="bj-note">Worst case per mode. Switch to the '
                'Thermal tab for the per-temperature detail.</div>',
                unsafe_allow_html=True)
    st.download_button("Download results CSV", df.to_csv(index=False),
                       file_name=f"boltjoint_{thread.replace('/', '-')}.csv",
                       mime="text/csv")

with tab_therm:
    st.markdown('<div class="bj-note">A tensile thermal load adds to maximum '
                'preload for the bolt checks; a preload-relieving load '
                'subtracts from minimum preload for separation. One '
                'temperature cannot drive both, so both extremes are run.'
                '</div>', unsafe_allow_html=True)
    st.dataframe(envelope.thermal_loads().style.format({
        "dT [F]": "{:+.1f}", "P_thermal [lbf]": "{:+,.2f}",
        "P_b [lbf]": "{:,.2f}", "governing MS": "{:,.3f}"}),
        width="stretch", hide_index=True)

    st.markdown('<div class="bj-h" style="font-size:.95rem">'
                'Layer-by-layer breakdown</div>', unsafe_allow_html=True)
    which = st.radio("Temperature case", ["hot", "cold"], horizontal=True)
    res = joint.thermal_breakdown(environment.cases[which])
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="bj-card-k">Free expansion, joint</div>'
                f'<div class="bj-card-v">{res.free_expansion_joint:+.3e} in</div>',
                unsafe_allow_html=True)
    m2.markdown(f'<div class="bj-card-k">Free expansion, bolt</div>'
                f'<div class="bj-card-v">{res.free_expansion_bolt:+.3e} in</div>',
                unsafe_allow_html=True)
    m3.markdown(f'<div class="bj-card-k">Mismatch &rarr; P_thermal</div>'
                f'<div class="bj-card-v">{res.P_thermal:+,.2f} lbf</div>',
                unsafe_allow_html=True)
    st.dataframe(res.to_dataframe().style.format({
        "thickness [in]": "{:.4f}", "d_start [in]": "{:.4f}",
        "E [psi]": "{:,.0f}", "alpha [ppm/F]": "{:.2f}",
        "k_i [lbf/in]": "{:,.0f}", "free expansion [in]": "{:.3e}",
        "constrained defl [in]": "{:.3e}"}), width="stretch", hide_index=True)

    if stiff_method == "frustum":
        st.markdown('<div class="bj-note">Cones start at both bearing faces '
                    'and expand toward the midplane, so the stack is split at '
                    'every material boundary and at the midplane. Stiffnesses '
                    'combine in series.</div>', unsafe_allow_html=True)

with tab_int:
    st.markdown('<div class="bj-note">TM-106943 eq. 56. The margin is the '
                'distance from the point to the curve; eq. 59 approximates it. '
                'The Guideline form is reported alongside because the two do '
                'not agree numerically.</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=140)
    Rs_c = np.linspace(0, 1, 300)
    ax.plot(Rs_c, np.sqrt(np.clip(1 - Rs_c ** 3, 0, None)), color=INK, lw=1.6,
            label="$R_t^2 + R_s^3 = 1$")
    ax.fill_between(Rs_c, np.sqrt(np.clip(1 - Rs_c ** 3, 0, None)),
                    color=BLUE, alpha=.06)
    rows = []
    for basis, colour in (("yield", CAUTION), ("ult", BLUE)):
        Rt, Rs = M.load_ratios(analysis.P_b, V, fastener, bolt_mat, basis,
                               sf_s, shank_shear)
        ax.plot(Rs, Rt, "o", color=colour, ms=9, mec="white", mew=1.4,
                label=f"{basis}  ({Rs:.3f}, {Rt:.3f})", zorder=5)
        rows.append({
            "basis": basis, "Rt": Rt, "Rs": Rs,
            "MS (TM-106943)": M.combined_margin(Rt, Rs, basis, method="tm106943").MS,
            "MOS (Guideline)": M.combined_margin(Rt, Rs, basis, method="guideline").MS})
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("shear load ratio  $R_s$"); ax.set_ylabel("tension load ratio  $R_t$")
    ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    plt.tight_layout()
    ci, ct = st.columns([1, 1])
    ci.pyplot(fig)
    ct.dataframe(pd.DataFrame(rows).style.format({
        "Rt": "{:.4f}", "Rs": "{:.4f}",
        "MS (TM-106943)": "{:.3f}", "MOS (Guideline)": "{:.3f}"}),
        width="stretch", hide_index=True)

with tab_sens:
    st.markdown('<div class="bj-note">Governing margin as one input is swept, '
                'everything else held. The preload sweep shows the tradeoff: '
                'more preload protects against separation and costs bolt '
                'tension margin.</div>', unsafe_allow_html=True)

    ax_sweep = model.sweep_load("P_et", np.linspace(0, max(4 * P_et, 40), 30))
    sh_sweep = model.sweep_load("V", np.linspace(0, max(4 * V, 40), 30))
    pl = model.sweep_preload(np.linspace(0.2 * preload.Po, 2.0 * preload.Po, 30))

    fig2, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=140)
    for ax_, d, xc, lbl in ((axes[0], ax_sweep, "P_et", "external axial load [lbf]"),
                            (axes[1], sh_sweep, "V", "external shear load [lbf]")):
        ax_.plot(d[xc], d["critical MS"], color=BLUE, lw=2)
        ax_.axhline(0, color=FAIL, ls="--", lw=1.2)
        ax_.axvline(P_et if xc == "P_et" else V, color=INK, ls=":", lw=1)
        ax_.set_xlabel(lbl); ax_.set_ylabel("governing MS")
    for col, colour in (("critical", BLUE), ("bolt tension (yield)", CAUTION),
                        ("separation", PASS)):
        axes[2].plot(pl["Po"], pl[col], color=colour, lw=2 if col == "critical" else 1.2,
                     label=col)
    axes[2].axhline(0, color=FAIL, ls="--", lw=1.2)
    axes[2].axvline(preload.Po, color=INK, ls=":", lw=1)
    axes[2].set_ylim(-1, min(10, max(pl["critical"].max() * 1.2, 1)))
    axes[2].set_xlabel("nominal preload $P_o$ [lbf]"); axes[2].set_ylabel("MS")
    axes[2].legend(fontsize=7, frameon=False)
    for ax_ in axes:
        ax_.grid(alpha=.25, lw=.6); ax_.set_facecolor(CARD)
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)
    fig2.patch.set_facecolor(CARD); plt.tight_layout()
    st.pyplot(fig2)

    good = ax_sweep[ax_sweep["critical MS"] >= 0]["P_et"]
    if len(good) < len(ax_sweep):
        st.markdown(f'<div class="bj-note">Largest swept axial load holding a '
                    f'positive margin: <b>{good.max():,.1f} lbf</b>.</div>',
                    unsafe_allow_html=True)

with tab_joint:
    jc, pc = st.columns(2)
    def _fmt(d: dict, places: int = 4):
        # Arrow cannot serialise a mixed str/float column, so format to text
        return pd.DataFrame(
            {"value": [f"{v:,.{places}f}" if isinstance(v, (int, float)) else str(v)
                       for v in d.values()]}, index=list(d))

    jc.dataframe(_fmt(joint.summary()), width="stretch")
    pc.dataframe(_fmt(preload.summary(), 3), width="stretch")
    st.text(analysis.report())

st.markdown(f"""<div class="bj-note" style="margin-top:2.2rem;border-top:1px solid {RULE};padding-top:.9rem">
Material properties are typical values for convenience. Replace them with MMPDS
and the applicable fastener procurement specification before releasing any
analysis. Bearing allowables default to 1.5&middot;Ftu. Fatigue, self-loosening
and fastener-pattern load distribution are not covered.
</div>""", unsafe_allow_html=True)
