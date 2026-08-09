"""Regression tests.  Each 'revb' test pins a bug found in the RevB notebook."""

import math

import pytest

from boltjoint import (
    Fastener, Joint, Layer, JointAnalysis, LoadCase, InsertSpec,
    materials, margins as M, preload_from_yield_fraction, preload_from_torque,
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_tensile_stress_area_2_56():
    """#2-56 At = 0.00370 in^2 (Machinery's Handbook)."""
    assert Fastener("2-56").At == pytest.approx(0.00370, abs=5e-5)


def test_tensile_stress_area_quarter_28():
    """1/4-28 At = 0.0364 in^2."""
    assert Fastener("1/4-28").At == pytest.approx(0.0364, abs=5e-4)


def test_revb_no_rounding_of_area():
    """RevB used np.round(At,3): 0.003696 -> 0.004, an 8% error."""
    At = Fastener("2-56").At
    assert abs(round(At, 3) - At) / At > 0.05
    assert At != round(At, 3)


def test_minor_diameter():
    f = Fastener("2-56")
    assert f.d_minor == pytest.approx(f.d_major - 1.082532 / f.tpi)


# ---------------------------------------------------------------------------
# The RevB area bug
# ---------------------------------------------------------------------------
def test_revb_thread_shear_area_is_not_tensile_area():
    """RevB used the thread shear area for tension: ~4.6x too large."""
    f = Fastener("2-56")
    Le = 1.5 * f.d_major
    As = M.bolt_thread_shear_area(f, Le)
    assert As / f.At > 4.0


def test_tension_margin_uses_At():
    f = Fastener("2-56")
    mat = materials.get("A286")
    r = M.tension_margin(100.0, f, mat, "ult")
    assert r.allowable == pytest.approx(mat.Ftu * f.At)


# ---------------------------------------------------------------------------
# Factor of safety handling
# ---------------------------------------------------------------------------
def test_sf_not_applied_to_preload():
    """Doubling SF must not double the preload contribution."""
    f = Fastener("2-56")
    bolt = materials.get("A286")
    al = materials.get("AL6061-T6")
    j = Joint(f, [Layer(0.060, al), Layer(0.129, al)], bolt,
              configuration=3, insert_length=0.129)
    p = preload_from_yield_fraction(f, bolt, 0.65, K=0.20, u=0.35)
    pb1 = M.bolt_axial_load(p, j, 10.0, SF=1.0)
    pb2 = M.bolt_axial_load(p, j, 10.0, SF=2.0)
    assert pb2 - pb1 == pytest.approx(j.n * j.phi * 10.0)
    assert pb2 < 2 * pb1


def test_max_preload_for_strength_min_for_separation():
    f = Fastener("2-56")
    bolt = materials.get("A286")
    al = materials.get("AL6061-T6")
    j = Joint(f, [Layer(0.060, al), Layer(0.129, al)], bolt,
              configuration=3, insert_length=0.129)
    p = preload_from_yield_fraction(f, bolt, 0.65)
    assert M.bolt_axial_load(p, j, 0.0, 1.4) == pytest.approx(p.Po_max)
    sep = M.separation_margin(p, j, 10.0)
    assert sep.allowable == pytest.approx(p.Po_min)
    assert p.Po_min < p.Po < p.Po_max


def test_preload_relaxation_applied():
    f = Fastener("2-56")
    p = preload_from_yield_fraction(f, materials.get("A286"), 0.65,
                                    u=0.35, relax=0.05)
    assert p.Po_min == pytest.approx(p.Po * (1 - 0.35) - 0.05 * p.Po)


# ---------------------------------------------------------------------------
# Interaction equations
# ---------------------------------------------------------------------------
def test_combined_pure_tension_matches_tension_margin():
    """With Rs=0 the combined margin reduces to 1/Rt - 1."""
    r = M.combined_margin(0.5, 0.0, "ult")
    assert r.MS == pytest.approx(1.0 / 0.5 - 1.0)


def test_combined_on_interaction_curve_is_zero():
    """Rt^2 + Rs^3 = 1 must give MS = 0."""
    Rs = 0.5
    Rt = math.sqrt(1.0 - Rs ** 3)
    assert M.combined_margin(Rt, Rs, "ult").MS == pytest.approx(0.0, abs=1e-9)
    assert M.combined_margin(Rt, Rs, "ult", method="guideline").MS == \
        pytest.approx(0.0, abs=1e-9)


def test_shear_yield_is_von_mises():
    mat = materials.get("A286")
    assert mat.Fsy == pytest.approx(0.577 * mat.Fty)


def test_shear_ultimate_not_075_ftu():
    """RevB used 0.75*Ftu; references say 0.55-0.60."""
    mat = materials.get("A286")
    assert 0.5 * mat.Ftu <= mat.Fsu <= 0.62 * mat.Ftu


# ---------------------------------------------------------------------------
# Insert pull-out
# ---------------------------------------------------------------------------
def test_pullout_takes_minimum_not_maximum():
    """RevB used np.max(P_ult), reporting the strongest mode."""
    f = Fastener("2-56")
    bolt = materials.get("A286")
    ins = materials.get("CRES304")
    parent = materials.get("AL6061-T6")
    res = M.insert_pullout_margins(72.0, f, bolt, ins, parent,
                                   Le_bolt_insert=0.129,
                                   insert_major_dia=0.1380,
                                   Le_insert_parent=0.129)
    gov = M.pullout_allowable(res)
    assert gov.allowable == min(r.allowable for r in res)
    assert gov.allowable < max(r.allowable for r in res)


def test_pullout_modes_use_different_areas():
    """Each mode must use its own mating geometry."""
    f = Fastener("2-56")
    mats = materials.get("A286"), materials.get("CRES304"), materials.get("AL6061-T6")
    res = M.insert_pullout_margins(72.0, f, *mats, 0.129, 0.1380, 0.150)
    allowables = [r.allowable for r in res]
    assert len(set(round(a, 6) for a in allowables)) == 3


# ---------------------------------------------------------------------------
# Stiffness / joint
# ---------------------------------------------------------------------------
def test_loading_plane_factor_bounded():
    al = materials.get("AL6061-T6")
    bolt = materials.get("A286")
    f = Fastener("2-56")
    j = Joint(f, [Layer(0.060, al), Layer(0.129, al)], bolt,
              configuration=3, insert_length=0.129)
    assert 0.0 <= j.n <= 1.0
    assert 0.0 < j.phi < 1.0


def test_stiffer_bolt_takes_more_load():
    al = materials.get("AL6061-T6")
    steel = materials.get("A286")
    ti = materials.get("Ti-6Al-4V")
    f = Fastener("2-56")
    layers = [Layer(0.060, al), Layer(0.129, al)]
    j_steel = Joint(f, layers, steel, configuration=3, insert_length=0.129)
    j_ti = Joint(f, layers, ti, configuration=3, insert_length=0.129)
    assert j_steel.phi > j_ti.phi


def test_thermal_sign_cold_relieves_preload():
    """Aluminium joint, steel bolt, cooling down -> preload loss."""
    al = materials.get("AL6061-T6")
    bolt = materials.get("A286")
    f = Fastener("2-56")
    j = Joint(f, [Layer(0.060, al), Layer(0.129, al)], bolt,
              configuration=3, insert_length=0.129)
    assert j.alpha_joint_eq > bolt.alpha
    assert j.thermal_load(-81.0) < 0.0
    assert j.thermal_load(+81.0) > 0.0


def test_configuration_head_style_guard():
    al = materials.get("AL6061-T6")
    bolt = materials.get("A286")
    with pytest.raises(ValueError):
        Joint(Fastener("2-56", head_style="flat100"),
              [Layer(0.060, al), Layer(0.129, al)], bolt,
              configuration=3, insert_length=0.129)


def test_insert_length_cannot_exceed_parent():
    al = materials.get("AL6061-T6")
    bolt = materials.get("A286")
    with pytest.raises(ValueError):
        Joint(Fastener("2-56"), [Layer(0.060, al), Layer(0.100, al)], bolt,
              configuration=3, insert_length=0.200)


# ---------------------------------------------------------------------------
# Bearing
# ---------------------------------------------------------------------------
def test_head_bearing_is_stress_not_product():
    """RevB computed P_b * A instead of P_b / A."""
    f = Fastener("2-56")
    al = materials.get("AL6061-T6")
    r = M.head_bearing_margin(100.0, al, f, hole_dia=0.096, washer_od=0.140)
    Abr = math.pi / 4 * (0.140 ** 2 - 0.096 ** 2)
    assert r.allowable == pytest.approx(al.Fbry * Abr)
    assert 100.0 / Abr > 1000.0            # a real stress, not ~0


def test_tearout_flags_low_edge_distance():
    f = Fastener("2-56")
    al = materials.get("AL6061-T6")
    r = M.shear_tearout_margin(10.0, al, 0.06, 0.100, f, 1.4)
    assert "1.5" in r.note or "2.0" in r.note


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def _demo_analysis(**lc_kwargs):
    f = Fastener("2-56")
    bolt = materials.get("A286")
    al = materials.get("AL6061-T6")
    j = Joint(f, [Layer(0.060, al), Layer(0.129, al)], bolt,
              configuration=3, insert_length=0.129)
    p = preload_from_yield_fraction(f, bolt, 0.65, K=0.20, u=0.35, relax=0.05)
    ins = InsertSpec(materials.get("CRES304"), al, 0.1380, 0.129, 0.129)
    return JointAnalysis(f, bolt, j, p, LoadCase(**lc_kwargs), insert=ins)


def test_end_to_end_runs_both_bases():
    a = _demo_analysis(P_et=4.0, V=2.0, delta_T=-81.0)
    bases = {r.basis for r in a.results}
    assert "yield" in bases and "ult" in bases
    assert a.critical.MS == min(r.MS for r in a.results)


def test_yield_margin_lower_than_ultimate():
    a = _demo_analysis(P_et=4.0, V=2.0)
    ms = {(r.mode, r.basis): r.MS for r in a.results}
    assert ms[("bolt tension", "yield")] < ms[("bolt tension", "ult")]


def test_higher_external_load_lowers_margin():
    a1 = _demo_analysis(P_et=4.0, V=2.0)
    a2 = _demo_analysis(P_et=40.0, V=2.0)
    assert a2.critical.MS < a1.critical.MS


def test_report_renders():
    assert "Critical:" in _demo_analysis(P_et=4.0, V=2.0).report()
