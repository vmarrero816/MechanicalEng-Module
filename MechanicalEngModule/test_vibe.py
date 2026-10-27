"""Regression tests for vibe.py -- each value is checked by hand or by statics."""

import math

import numpy as np
import pytest

from MechanicalEngModule import (
    Fastener, Joint, Layer, materials, margins as M, preload_from_yield_fraction,
)
from MechanicalEngModule import vibe as V

GEVS_F = [20, 50, 800, 2000]
GEVS_ASD = [0.026, 0.16, 0.16, 0.026]


# ---------------------------------------------------------------------------
# Natural frequency
# ---------------------------------------------------------------------------
def test_fn_spring_mass_static_deflection():
    """fn = (1/2pi) sqrt(g/delta): 0.01 in sag -> 31.27 Hz."""
    W, delta = 10.0, 0.01
    assert V.fn_spring_mass(W / delta, W) == pytest.approx(31.27, abs=0.01)


def test_fn_rayleigh_single_mass_equals_spring_mass():
    k, W = 5000.0, 4.0
    assert V.fn_rayleigh([W], [W / k]) == pytest.approx(V.fn_spring_mass(k, W))


def test_fn_cantilever_uses_3EI_over_L3_and_33_140_beam_mass():
    E, I, L, Wt, Wb = 10e6, 0.01, 5.0, 2.0, 1.4
    k = 3 * E * I / L ** 3
    assert V.fn_cantilever_tip_mass(E, I, L, Wt, Wb) == pytest.approx(
        V.fn_spring_mass(k, Wt + 0.2357 * Wb), rel=1e-3)


# ---------------------------------------------------------------------------
# Input environment
# ---------------------------------------------------------------------------
def test_grms_gevs_qual_is_14_1():
    assert V.grms(GEVS_F, GEVS_ASD) == pytest.approx(14.1, abs=0.1)


def test_grms_flat_profile_exact():
    assert V.grms([100, 1000], [0.1, 0.1]) == pytest.approx(math.sqrt(0.1 * 900))


def test_grms_minus_3db_branch():
    """ASD ~ 1/f is the m = -1 special case: area = a1 f1 ln(f2/f1)."""
    assert V.grms([10, 100], [1.0, 0.1]) == pytest.approx(math.sqrt(10 * math.log(10)))


def test_asd_slope_point():
    assert V.asd_slope_point(1.0, 1.0, 2.0, 10 * math.log10(2)) == pytest.approx(2.0)
    assert V.asd_slope_point(0.026, 20, 50, 6.0) == pytest.approx(0.16, rel=0.01)


def test_asd_at_breakpoints_loglog_and_out_of_band():
    assert V.asd_at(GEVS_F, GEVS_ASD, 800) == pytest.approx(0.16)
    assert V.asd_at([10, 1000], [1.0, 0.01], 100) == pytest.approx(0.1)   # log-log
    assert V.asd_at(GEVS_F, GEVS_ASD, 10) == 0.0                          # no input
    assert V.asd_at(GEVS_F, GEVS_ASD, 3000) == 0.0


def test_profile_validation():
    with pytest.raises(ValueError):
        V.grms([100, 50], [0.1, 0.1])


# ---------------------------------------------------------------------------
# Random and sine response
# ---------------------------------------------------------------------------
def test_miles_hand_value():
    """3*sqrt(pi/2 * 100 * 10 * 0.1) = 37.60 g."""
    assert V.miles_3sigma(100, 10, 0.1) == pytest.approx(37.60, abs=0.01)


def test_miles_envelope_finds_top_of_plateau():
    fn, g = V.miles_envelope(GEVS_F, GEVS_ASD, Q=10, fn_range=(100, 500))
    assert fn == pytest.approx(500, rel=1e-6)
    assert g == pytest.approx(V.miles_3sigma(500, 10, 0.16))


def test_miles_envelope_on_rising_slope_beats_point_estimate():
    """fn uncertain on the +6 dB/oct ramp: the envelope must not under-predict."""
    fn, g = V.miles_envelope(GEVS_F, GEVS_ASD, Q=10, fn_range=(25, 45))
    assert g >= V.miles_3sigma(35, 10, V.asd_at(GEVS_F, GEVS_ASD, 35))
    assert fn == pytest.approx(45, rel=1e-6)


def test_sine_resonant_and_rigid():
    f, A = [5, 100], [2.5, 2.5]
    assert V.sine_response(60, f, A, Q=10) == pytest.approx(25.0)
    assert V.sine_response(150, f, A, Q=10) == pytest.approx(2.5)


def test_sine_envelope_catches_band_overlap():
    """fn estimate above the band, but its range dips into it."""
    f, A = [5, 100], [2.5, 2.5]
    assert V.sine_response(110, f, A, Q=10) == pytest.approx(2.5)
    assert V.sine_envelope(f, A, Q=10, fn_range=(90, 130))[1] == pytest.approx(25.0)


def test_limit_g_envelopes_not_sums():
    assert V.limit_g(30.0, 25.0, 10.0) == 30.0


# ---------------------------------------------------------------------------
# Interface loads
# ---------------------------------------------------------------------------
def test_lateral_load_moment_sign():
    """+x load at height h gives My = +h*W*g, which lifts the -x side."""
    L = V.interface_loads(5.0, [20, 0, 0], [0, 0, 2.0])
    assert L.F == pytest.approx([100, 0, 0])
    assert L.M == pytest.approx([0, 200, 0])
    pat = V.BoltPattern([-1, 1, 1, -1], [-1, -1, 1, 1])
    b = V.bolt_loads(pat, L)
    assert np.all(b.P[pat.dx < 0] > 0) and np.all(b.P[pat.dx > 0] < 0)


def test_axis_loads_one_axis_at_a_time():
    xs = V.axis_interface_loads(2.0, [0, 0, 1.0], 10, 20, 30)
    assert [c.name for c in xs] == ["X", "Y", "Z"]
    assert xs[2].F == pytest.approx([0, 0, 60])


# ---------------------------------------------------------------------------
# Bolt distribution
# ---------------------------------------------------------------------------
def test_bolt_loads_statics_irregular_pattern():
    """Any pattern, any load: bolt forces must balance all six components."""
    pat = V.BoltPattern([0.0, 3.1, 4.0, 0.7, 2.2], [0.0, 0.4, 2.9, 2.1, 1.0])
    L = V.InterfaceLoad(np.array([40.0, -25.0, 130.0]),
                        np.array([210.0, -95.0, 60.0]))
    b = V.bolt_loads(pat, L)
    dx, dy = pat.dx, pat.dy
    assert abs(pat.Ixy) > 0.1
    assert b.P.sum() == pytest.approx(130.0)
    assert (b.P * dy).sum() == pytest.approx(210.0)
    assert (b.P * dx).sum() == pytest.approx(95.0)          # = -My
    assert b.Vx.sum() == pytest.approx(40.0)
    assert b.Vy.sum() == pytest.approx(-25.0)
    assert (dx * b.Vy - dy * b.Vx).sum() == pytest.approx(60.0)


def test_bolt_circle_moment_matches_2M_over_nR():
    R, Mx = 2.0, 400.0
    pat = V.BoltPattern.circle(4, R, start_deg=90)
    b = V.bolt_loads(pat, V.InterfaceLoad(np.zeros(3), np.array([Mx, 0, 0])))
    assert b.P.max() == pytest.approx(2 * Mx / (4 * R))


def test_torsion_shear_on_bolt_circle():
    R, Mz, n = 1.5, 300.0, 6
    pat = V.BoltPattern.circle(n, R)
    b = V.bolt_loads(pat, V.InterfaceLoad(np.zeros(3), np.array([0, 0, Mz])))
    assert b.V == pytest.approx(np.full(n, Mz / (n * R)))


def test_collinear_pattern_cannot_react_moment():
    pat = V.BoltPattern([0, 1, 2], [0, 0, 0])
    with pytest.raises(ValueError):
        V.bolt_loads(pat, V.InterfaceLoad(np.zeros(3), np.array([50.0, 0, 0])))


def test_prying_scales_tension_only():
    pat = V.BoltPattern([-1, 1], [0, 0])
    L = V.InterfaceLoad(np.zeros(3), np.array([0, 100.0, 0]))
    b1, b2 = V.bolt_loads(pat, L), V.bolt_loads(pat, L, prying=1.3)
    t = b1.P > 0
    assert b2.P[t] == pytest.approx(1.3 * b1.P[t])
    assert b2.P[~t] == pytest.approx(b1.P[~t])


def test_rectangle_pattern_counts_perimeter_once():
    assert V.BoltPattern.rectangle(3, 3, 1.0, 1.0).n == 8


def test_critical_reports_both_bolts():
    pat = V.BoltPattern([-1, 1, 1, -1], [-1, -1, 1, 1])
    L = V.interface_loads(3.0, [30, 0, 0], [0.5, 0, 1.5])
    c = V.bolt_loads(pat, L).critical()
    assert set(c) == {"max_tension", "max_shear"}
    assert c["max_tension"][1] > 0


# ---------------------------------------------------------------------------
# Joint checks
# ---------------------------------------------------------------------------
@pytest.fixture
def joint_10_32():
    f = Fastener("10-32")
    bolt = materials.get("A286")
    al = materials.get("AL6061-T6")
    j = Joint(f, [Layer(0.125, al), Layer(0.250, al)], bolt,
              configuration=3, insert_length=0.250)
    p = preload_from_yield_fraction(f, bolt, 0.65, K=0.20, u=0.35)
    return f, bolt, j, p


def test_run_vibe_case_uses_basis_factors(joint_10_32):
    f, bolt, j, p = joint_10_32
    r = V.run_vibe_case(f, bolt, j, p, P_et=150.0, V=60.0,
                        FS_y=1.25, FS_u=1.4, FF_y=1.0, FF_u=1.15)
    dPb = r.ult_run.P_b - r.yield_run.P_b
    assert dPb == pytest.approx(j.n * j.phi * (1.15 * 1.4 - 1.25) * 150.0)
    assert {x.basis for x in r.results} == {"yield", "ult", "n/a"}
    assert sum(x.mode == "joint separation" for x in r.results) == 1
    assert sum(x.mode == "bolt tension" for x in r.results) == 2


def test_run_vibe_case_separation_carries_fitting_factor(joint_10_32):
    f, bolt, j, p = joint_10_32
    r = V.run_vibe_case(f, bolt, j, p, P_et=150.0, V=0.0, FS_sep=1.2, FF_sep=1.15)
    sep = next(x for x in r.results if x.mode == "joint separation")
    assert sep.SF == pytest.approx(1.2 * 1.15)


def test_slip_margin_nasa_std_5020_A10_6(joint_10_32):
    *_, p = joint_10_32
    mu, FS, FF, Ps, Pt = 0.1, 1.25, 1.0, 40.0, 100.0
    r = V.slip_margin(Ps, Pt, p, mu, FS, FF)
    assert r.MS == pytest.approx(mu * p.Po_min / (FF * FS * (Ps + mu * Pt)) - 1)


def test_slip_thermal_relief_reduces_margin(joint_10_32):
    *_, p = joint_10_32
    a = V.slip_margin(40.0, 0.0, p, 0.1, 1.25)
    b = V.slip_margin(40.0, 0.0, p, 0.1, 1.25, P_thermal=-50.0)
    assert b.MS < a.MS


# ---------------------------------------------------------------------------
# Fatigue
# ---------------------------------------------------------------------------
def test_steinberg_hand_value():
    """fn 200 Hz, 120 s, LF 4, b 6.4, S-N through (10 ksi, 1e6)."""
    S1, fn, T, b = 5000.0, 200.0, 120.0, 6.4
    n = fn * 4 * T * np.array([0.683, 0.271, 0.0433])
    N = 1e6 * (10000.0 / (S1 * np.array([1, 2, 3]))) ** b
    assert V.steinberg_damage(S1, fn, T, b, 10000.0, 1e6) == pytest.approx((n / N).sum())


def test_steinberg_zero_stress_and_linear_in_time():
    assert V.steinberg_damage(0.0, 200, 60, 6, 1e4, 1e6) == 0.0
    d1 = V.steinberg_damage(3000, 200, 60, 6, 1e4, 1e6)
    assert V.steinberg_damage(3000, 200, 120, 6, 1e4, 1e6) == pytest.approx(2 * d1)


def test_bolt_alternating_stress_uses_n_phi(joint_10_32):
    f, _, j, _ = joint_10_32
    assert V.bolt_alternating_stress(30.0, j, f) == pytest.approx(j.n * j.phi * 30 / f.At)


def test_low_risk_screen():
    ok = dict(is_A286=True, thread_form="UNJ", rolled_after_heat_treat=True,
              P_tL=200.0, P_tu_allow=1000.0, fatigue_factor=1.0,
              E_clamped_min=10e6, E_bolt=29e6, edge_distance=0.3, d_major=0.19,
              Pp_min=500.0)
    assert V.low_risk_fatigue_screen(**ok)["LOW RISK"]
    assert not V.low_risk_fatigue_screen(**{**ok, "thread_form": "UNR",
                                            "P_tL": 450.0})["LOW RISK"]
    assert not V.low_risk_fatigue_screen(**{**ok, "edge_distance": 0.2})["LOW RISK"]


# ---------------------------------------------------------------------------
# Test verification
# ---------------------------------------------------------------------------
def test_half_power_recovers_Q_of_sdof():
    zeta, fn = 0.05, 150.0
    f = np.linspace(50, 300, 20001)
    r = f / fn
    T = np.sqrt((1 + (2 * zeta * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2))
    f_pk, Q = V.q_half_power(f, T)
    assert f_pk == pytest.approx(fn, rel=0.01)
    assert Q == pytest.approx(1 / (2 * zeta), rel=0.05)


def test_signature_shift():
    shift, ok = V.signature_shift(200.0, 188.0)
    assert shift == pytest.approx(-0.06) and not ok
    assert V.signature_shift(200.0, 196.0)[1]
