"""Tests for case.py: the shared notebook/GUI case builder."""

import math

import numpy as np
import pandas as pd
import pytest

from MechanicalEngModule import (
    AbutmentSpec, Fastener, InsertSpec, Joint, JointAnalysis, JointCase, Layer,
    LoadCase, NUT_FACTOR, Preload, ThermalEnvironment, UNCERTAINTY, materials,
    preload_target, run_thermal_envelope,
)


def _manual_model():
    """The Rev C example built by hand, the way the notebook used to."""
    f = Fastener("2-56", head_style="socket")
    bolt = materials.get("A286")
    al = materials.get("AL6061-T6")
    layers = [Layer(0.060, al), Layer(0.129, al)]
    Li = 1.5 * f.d_major
    j = Joint(f, layers, bolt, configuration=3, insert_length=Li,
              stiffness_method="frustum", cone_angle_deg=30.0)
    env = ThermalEnvironment(20.0, 71.0, -25.0)
    p = preload_target(81.0, f, K=NUT_FACTOR["dry"],
                       u=UNCERTAINTY["torque wrench, unlubricated"], relax=0.05)
    lc = LoadCase("case 1", P_et=4.0, V=2.0, Mb=0.0, delta_T=env.cases["hot"],
                  SF_tension=1.4, SF_shear=1.4, SF_separation=1.2)
    ins = InsertSpec(materials.get("CRES304"), al, 0.1380, Li, 0.129)
    ab = AbutmentSpec(al, 0.060, 0.200, 0.096, 0.096)
    return f, bolt, j, p, lc, env, ins, ab


def test_json_round_trip(tmp_path):
    c = JointCase(thread="10-32", layers=[[0.125, "AL7075-T73"], [0.25, "AL6061-T6"]],
                  P_et=150.0, washer_od=None)
    path = tmp_path / "case.json"
    c.to_json(path)
    assert JointCase.from_json(path) == c
    assert JointCase.from_json(c.to_json()) == c


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown case fields"):
        JointCase.from_dict({"thread": "2-56", "P_ET": 4.0})


def test_bad_preload_method_rejected():
    with pytest.raises(ValueError):
        JointCase(preload_method="guess").build()


def test_build_matches_manual_construction():
    f, bolt, j, p, lc, env, ins, ab = _manual_model()
    manual = JointAnalysis(f, bolt, j, p, lc, insert=ins, abutment=ab)
    shared = JointCase().build().analysis("hot")
    assert [(r.mode, r.basis) for r in shared.results] == \
           [(r.mode, r.basis) for r in manual.results]
    for a, b in zip(shared.results, manual.results):
        assert a.MS == pytest.approx(b.MS)
    assert shared.P_b == pytest.approx(manual.P_b)


def test_envelope_matches_manual():
    f, bolt, j, p, lc, env, ins, ab = _manual_model()
    manual = run_thermal_envelope(f, bolt, j, p, lc, env, insert=ins, abutment=ab)
    shared = JointCase().build().envelope()
    assert shared.critical.MS == pytest.approx(manual.critical.MS)
    assert shared.critical_case == manual.critical_case


def test_worst_agrees_with_groupby_envelope():
    """envelope() now comes from worst(); check it against the old groupby form."""
    env = JointCase().build().envelope()
    frames = []
    for name, a in env.cases.items():
        df = a.to_dataframe()[["mode", "basis", "MS", "pass"]].copy()
        df["case"] = name
        frames.append(df)
    allc = pd.concat(frames, ignore_index=True)
    old = allc.loc[allc.groupby(["mode", "basis"])["MS"].idxmin()]
    old = old.set_index(["mode", "basis"])["MS"].sort_index()
    new = env.envelope().set_index(["mode", "basis"])["MS"].sort_index()
    pd.testing.assert_series_equal(new, old)
    ms = [r.MS for r, _ in env.worst()]
    assert ms == sorted(ms)


def test_preload_methods():
    base = JointCase()
    f = Fastener(base.thread)
    K = NUT_FACTOR[base.install]
    assert JointCase(preload_method="torque", applied_torque=2.1).build().preload.Po \
        == pytest.approx(2.1 / (K * f.d_major))
    y = JointCase(preload_method="yield_fraction", yield_fraction=0.65).build()
    assert y.preload.Po == pytest.approx(0.65 * materials.get("A286").Fty * f.At)


def test_no_insert_for_through_bolt():
    m = JointCase(configuration=1).build()
    assert m.insert is None and m.Li is None


def test_abutment_optional():
    assert JointCase(use_abutment=False).build().abutment is None


def test_sweep_load_matches_direct_runs():
    m = JointCase().build()
    vals = np.linspace(0, 40, 5)
    sw = m.sweep_load("P_et", vals)
    for v, ms in zip(vals, sw["critical MS"]):
        assert ms == pytest.approx(m.analysis("hot", P_et=v).critical.MS)
    with pytest.raises(ValueError):
        m.sweep_load("delta_T", vals)


def test_sweep_preload_columns_and_trend():
    m = JointCase().build()
    sw = m.sweep_preload(np.linspace(20, 160, 8))
    assert list(sw.columns) == ["Po", "critical", "bolt tension (yield)", "separation"]
    assert sw["separation"].is_monotonic_increasing          # more preload helps
    assert sw["bolt tension (yield)"].is_monotonic_decreasing  # and costs tension


def test_warnings():
    short = JointCase(engagement_dia=0.5).build()
    assert any("below the equal-strength" in w for w in short.warnings())
    edge = JointCase(edge_distance=0.1).build()
    assert any("e/D" in w for w in edge.warnings())
