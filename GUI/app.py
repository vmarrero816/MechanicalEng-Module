"""Bolted joint analysis - interactive front end.

    streamlit run app.py

All physics comes from the boltjoint package; this module only collects inputs
and renders results.  No calculation lives here.
"""

from __future__ import annotations

import io
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boltjoint import (  # noqa: E402
    AbutmentSpec, Fastener, InsertSpec, Joint, JointAnalysis, Layer, LoadCase,
    Preload, materials, margins as M, min_engagement_equal_strength,
    preload_from_torque, preload_from_yield_fraction, preload_target,
    NUT_FACTOR, UNCERTAINTY,
)
from boltjoint.fasteners import THREADS  # noqa: E402

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
sb.markdown("## Fastener")
thread = sb.selectbox("Thread", sorted(THREADS), index=sorted(THREADS).index("2-56"))
head_style = sb.selectbox("Head style", ["socket", "flat100"])
bolt_key = sb.selectbox("Bolt material", MAT_KEYS, index=MAT_KEYS.index("A286"))
reduced = sb.checkbox("Reduced shank")
shank_dia = sb.number_input("Shank diameter [in]", 0.01, 1.0,
                            float(THREADS[thread][0]), format="%.4f") if reduced else None
shank_shear = sb.checkbox("Shear plane on shank", value=False,
                          help="Otherwise the threads are in the shear plane and At is used.")

sb.markdown("## Joint stack")
config = sb.selectbox("Configuration", [1, 2, 3, 4], index=2, format_func=lambda c: {
    1: "1 - through-bolt + nut",
    2: "2 - through-bolt + nut, flat head",
    3: "3 - insert / tapped hole",
    4: "4 - insert / tapped hole, flat head"}[c])
n_layers = sb.number_input("Number of layers", 2, 5, 2)
layers_in = []
for i in range(int(n_layers)):
    last = i == n_layers - 1
    tag = "parent" if (last and config in (3, 4)) else f"layer {i + 1}"
    c1, c2 = sb.columns([1, 1.3])
    t = c1.number_input(f"t {tag} [in]", 0.001, 5.0, 0.129 if last else 0.060,
                        format="%.4f", key=f"t{i}")
    m = c2.selectbox(f"mat {tag}", MAT_KEYS, index=MAT_KEYS.index("AL6061-T6"), key=f"m{i}")
    layers_in.append((t, m))
engage_dia = sb.number_input("Engagement [diameters]", 0.5, 4.0, 1.5, 0.1) \
    if config in (3, 4) else None

sb.markdown("## Preload")
method = sb.radio("Basis", ["target force", "applied torque", "fraction of yield"])
target_P = sb.number_input("Target preload [lbf]", 0.0, 1e5, 81.0) if method == "target force" else None
applied_T = sb.number_input("Applied torque [lbf-in]", 0.0, 1e5, 2.1) if method == "applied torque" else None
yield_frac = sb.slider("Fraction of bolt yield", 0.2, 0.9, 0.65, 0.01) if method == "fraction of yield" else None
install = sb.selectbox("Installation", sorted(NUT_FACTOR), index=sorted(NUT_FACTOR).index("dry"))
unc_key = sb.selectbox("Preload uncertainty", sorted(UNCERTAINTY),
                       index=sorted(UNCERTAINTY).index("torque wrench, unlubricated"))
relax = sb.slider("Short-term relaxation", 0.0, 0.15, 0.05, 0.01)

sb.markdown("## Applied loads")
P_et = sb.number_input("Axial load P_et [lbf]", 0.0, 1e6, 4.0)
V = sb.number_input("Shear load V [lbf]", 0.0, 1e6, 2.0)
Mb = sb.number_input("Bending moment [lbf-in]", 0.0, 1e6, 0.0)
T_asm = sb.number_input("Assembly temperature [C]", -273.0, 500.0, 20.0)
T_ext = sb.number_input("Extreme temperature [C]", -273.0, 500.0, -25.0)
sf_t = sb.number_input("SF tension", 1.0, 5.0, 1.4, 0.05)
sf_s = sb.number_input("SF shear", 1.0, 5.0, 1.4, 0.05)
sf_sep = sb.number_input("SF separation", 1.0, 3.0, 1.2, 0.05,
                         help="1.2 structural, 1.4 pressure systems")

use_insert = config in (3, 4)
if use_insert:
    sb.markdown("## Threaded insert")
    ins_key = sb.selectbox("Insert material", MAT_KEYS, index=MAT_KEYS.index("CRES304"))
    ins_major = sb.number_input("Insert external major dia [in]", 0.01, 2.0, 0.1380, format="%.4f")
    le_ins_par = sb.number_input("Insert-into-parent engagement [in]", 0.01, 2.0, 0.129, format="%.4f")

sb.markdown("## Abutment")
use_ab = sb.checkbox("Check tear-out and bearing", value=True)
if use_ab:
    ab_key = sb.selectbox("Abutment material", MAT_KEYS, index=MAT_KEYS.index("AL6061-T6"))
    sheet_t = sb.number_input("Sheet thickness [in]", 0.005, 2.0, 0.060, format="%.4f")
    edge_d = sb.number_input("Edge distance e [in]", 0.01, 5.0, 0.200, format="%.4f")
    hole_d = sb.number_input("Through-hole dia [in]", 0.01, 2.0, 0.096, format="%.4f")
    washer_od = sb.number_input("Washer OD [in], 0 = none", 0.0, 3.0, 0.096, format="%.4f")

n_bolts = sb.number_input("Bolts in assembly (reporting)", 1, 500, 7)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
try:
    fastener = Fastener(thread, head_style=head_style, shank_dia=shank_dia)
    bolt_mat = materials.get(bolt_key)
    layers = [Layer(t, materials.get(m)) for t, m in layers_in]
    Li = engage_dia * fastener.d_major if config in (3, 4) else None
    joint = Joint(fastener, layers, bolt_mat, configuration=config, insert_length=Li)

    K, u = NUT_FACTOR[install], UNCERTAINTY[unc_key]
    if method == "target force":
        preload = preload_target(target_P, fastener, K=K, u=u, relax=relax)
    elif method == "applied torque":
        preload = preload_from_torque(applied_T, K, fastener.d_major, u=u, relax=relax)
    else:
        preload = preload_from_yield_fraction(fastener, bolt_mat, yield_frac,
                                              K=K, u=u, relax=relax)

    load_case = LoadCase("case 1", P_et=P_et, V=V, Mb=Mb,
                         delta_T=1.8 * (T_ext - T_asm),
                         SF_tension=sf_t, SF_shear=sf_s, SF_separation=sf_sep)
    insert = InsertSpec(materials.get(ins_key), layers[-1].material,
                        ins_major, Li, le_ins_par) if use_insert else None
    abutment = AbutmentSpec(materials.get(ab_key), sheet_t, edge_d, hole_d,
                            washer_od or None) if use_ab else None

    analysis = JointAnalysis(fastener, bolt_mat, joint, preload, load_case,
                             insert=insert, abutment=abutment,
                             shear_plane_in_shank=shank_shear)
except (ValueError, KeyError) as exc:
    st.markdown(f'<div class="bj-err"><b>Cannot build the joint.</b><br>{exc}</div>',
                unsafe_allow_html=True)
    st.stop()

crit = analysis.critical
ok = analysis.passes
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
      <div class="bj-tb-key">{crit.basis}</div></div>
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
        f'<div class="bj-stamp-note">governing &mdash; {crit.mode} ({crit.basis})</div>',
        unsafe_allow_html=True)
    Le_min = min_engagement_equal_strength(fastener)
    if Li and Li < Le_min:
        st.markdown(f'<div class="bj-warn">Engagement {Li:.4f} in is below the '
                    f'equal-strength minimum {Le_min:.4f} in.</div>', unsafe_allow_html=True)
    if use_ab and edge_d / fastener.d_major < 1.5:
        st.markdown(f'<div class="bj-warn">e/D = {edge_d / fastener.d_major:.2f} '
                    f'&mdash; below 1.5. Use Bruhn / TM X-73305.</div>', unsafe_allow_html=True)
    if abs(crit.MS) < 0.5:
        st.markdown('<div class="bj-warn">Governing margin below 0.5. Check the '
                    'interaction curve directly; the approximation can be '
                    'unconservative.</div>', unsafe_allow_html=True)

with right:
    cards = [
        ("Preload Po", f"{preload.Po:,.1f} lbf",
         f"max {preload.Po_max:,.1f} / min {preload.Po_min:,.1f}"),
        ("Thermal load", f"{analysis.P_thermal:+,.1f} lbf",
         f"dT {load_case.delta_T:+.0f} F"),
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
st.markdown('<div class="bj-note">Every mode, both strength bases. The red line '
            'is MS = 0; the scale compresses with increasing margin so the '
            'design-driving band near zero stays readable.</div>', unsafe_allow_html=True)

# saturating scale f(ms) = ms/(1+ms): 0 -> 0, 0.5 -> .33, 1 -> .5, 3 -> .75, inf -> 1
NEG, POS = 14.0, 86.0          # percent of track width either side of zero


def _pos(ms: float) -> float:
    return POS * (ms / (1.0 + ms)) if ms != float("inf") else POS


TICKS = [(0.5, "0.5"), (1.0, "1"), (3.0, "3"), (10.0, "10")]
tick_html = "".join(
    f'<div class="bj-tick" style="left:{NEG + _pos(v):.2f}%"></div>' for v, _ in TICKS)

rungs = []
for r in sorted(analysis.results, key=lambda x: x.MS):
    ms = r.MS
    colour = FAIL if ms < 0 else (CAUTION if ms < 0.5 else PASS)
    gov = " bj-gov" if (r.mode, r.basis) == (crit.mode, crit.basis) else ""
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
        f'<div class="bj-rung-label">{r.mode} <small>{r.basis}</small></div>'
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
tab_tab, tab_int, tab_sens, tab_joint = st.tabs(
    ["Results table", "Interaction", "Sensitivity", "Joint properties"])

with tab_tab:
    df = analysis.to_dataframe()
    st.dataframe(
        df.style.apply(
            lambda row: [f"background-color: #FDF0EF" if not row["pass"] else ""
                         for _ in row], axis=1
        ).format({"applied": "{:,.2f}", "allowable": "{:,.2f}",
                  "SF": "{:.2f}", "MS": "{:,.3f}"}),
        width="stretch", height=560)
    st.download_button("Download results CSV", df.to_csv(index=False),
                       file_name=f"boltjoint_{thread.replace('/', '-')}.csv",
                       mime="text/csv")

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

    def rerun(preload_obj, lc):
        return JointAnalysis(fastener, bolt_mat, joint, preload_obj, lc,
                             insert=insert, abutment=abutment,
                             shear_plane_in_shank=shank_shear)

    def sweep_load(attr, values):
        out = []
        for v in values:
            lc = LoadCase(**{**load_case.__dict__, attr: float(v)})
            out.append({attr: v, "MS": rerun(preload, lc).critical.MS})
        return pd.DataFrame(out)

    ax_sweep = sweep_load("P_et", np.linspace(0, max(4 * P_et, 40), 30))
    sh_sweep = sweep_load("V", np.linspace(0, max(4 * V, 40), 30))
    pl_rows = []
    for Po in np.linspace(0.2 * preload.Po, 2.0 * preload.Po, 30):
        p = Preload(Po=Po, u=preload.u, relax=preload.relax,
                    K=preload.K, d_major=preload.d_major)
        a = rerun(p, load_case)
        by = {(r.mode, r.basis): r.MS for r in a.results}
        pl_rows.append({"Po": Po, "critical": a.critical.MS,
                        "bolt tension (yield)": by[("bolt tension", "yield")],
                        "separation": by[("joint separation", "n/a")]})
    pl = pd.DataFrame(pl_rows)

    fig2, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=140)
    for ax_, d, xc, lbl in ((axes[0], ax_sweep, "P_et", "external axial load [lbf]"),
                            (axes[1], sh_sweep, "V", "external shear load [lbf]")):
        ax_.plot(d[xc], d["MS"], color=BLUE, lw=2)
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

    good = ax_sweep[ax_sweep["MS"] >= 0]["P_et"]
    if len(good) < len(ax_sweep):
        st.markdown(f'<div class="bj-note">Largest swept axial load holding a '
                    f'positive margin: <b>{good.max():,.1f} lbf</b>.</div>',
                    unsafe_allow_html=True)

with tab_joint:
    jc, pc = st.columns(2)
    jc.dataframe(pd.Series(joint.summary()).to_frame("value")
                 .style.format("{:,.4f}"), width="stretch")
    pc.dataframe(pd.Series(preload.summary()).to_frame("value")
                 .style.format("{:,.3f}"), width="stretch")
    st.text(analysis.report())

st.markdown(f"""<div class="bj-note" style="margin-top:2.2rem;border-top:1px solid {RULE};padding-top:.9rem">
Material properties are typical values for convenience. Replace them with MMPDS
and the applicable fastener procurement specification before releasing any
analysis. Bearing allowables default to 1.5&middot;Ftu. Fatigue, self-loosening
and fastener-pattern load distribution are not covered.
</div>""", unsafe_allow_html=True)
