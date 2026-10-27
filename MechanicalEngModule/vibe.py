"""Random and sine vibration loads for preloaded joints -- hand-calc path.

Units: in, lbf, lbf-in, psi, Hz, g^2/Hz.  Weight W (lbf) stands in for mass,
so an inertial load is simply W * g-level.

No FEM model is assumed.  The natural frequency and Q are estimates, so each
step leans conservative.  Call order (see the "Vibe-to-Bolt Load Method" doc):

    1  fn estimate          fn_spring_mass, fn_cantilever_tip_mass, fn_rayleigh
    2  input environment    grms, asd_at, asd_slope_point
    3  random response      miles_3sigma, miles_envelope
    4  sine response        sine_response, sine_envelope
    5  limit g per axis     limit_g
    6  interface F/M        interface_loads, axis_interface_loads
    7  bolt loads           BoltPattern, bolt_loads
    8  joint checks         run_vibe_case (wraps JointAnalysis), slip_margin
    9  fatigue              bolt_alternating_stress, steinberg_damage,
                            low_risk_fatigue_screen
   10  test verification    q_half_power, signature_shift
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import margins as M
from .analysis import JointAnalysis, LoadCase

__all__ = [
    "G_IN", "STEINBERG_FRACTIONS",
    "fn_spring_mass", "fn_cantilever_tip_mass", "fn_rayleigh",
    "grms", "asd_slope_point", "asd_at",
    "miles_3sigma", "miles_envelope", "sine_response", "sine_envelope",
    "limit_g",
    "InterfaceLoad", "interface_loads", "axis_interface_loads",
    "BoltPattern", "BoltLoads", "bolt_loads",
    "VibeJointResult", "run_vibe_case", "slip_margin",
    "bolt_alternating_stress", "steinberg_cycles", "steinberg_damage",
    "low_risk_fatigue_screen", "q_half_power", "signature_shift",
]

G_IN = 386.089                               # standard gravity, in/s^2
STEINBERG_FRACTIONS = (0.683, 0.271, 0.0433)  # share of cycles at 1, 2, 3 sigma


# ---------------------------------------------------------------------------
# 1. Natural frequency estimates
# ---------------------------------------------------------------------------
def fn_spring_mass(k: float, W: float) -> float:
    """Single-DOF natural frequency, Hz.  fn = (1/2pi) sqrt(k g / W).

    k : support stiffness in the excitation axis, lbf/in
    W : supported weight, lbf
    """
    if k <= 0 or W <= 0:
        raise ValueError("k and W must be positive")
    return math.sqrt(k * G_IN / W) / (2.0 * math.pi)


def fn_cantilever_tip_mass(E: float, I: float, L: float, W_tip: float,
                           W_beam: float = 0.0) -> float:
    """Cantilever with a tip weight, Hz.

    k = 3EI/L^3 and W_eff = W_tip + (33/140) W_beam (Rayleigh's beam-mass
    correction).  E psi, I in^4, L in, weights lbf.
    """
    k = 3.0 * E * I / L ** 3
    return fn_spring_mass(k, W_tip + 33.0 / 140.0 * W_beam)


def fn_rayleigh(W, y) -> float:
    """Rayleigh's method for lumped weights, Hz.

        fn = (1/2pi) sqrt( g * sum(W_i y_i) / sum(W_i y_i^2) )

    W : lumped weights, lbf
    y : static deflections at those points under the weights applied in the
        excitation axis, in (same sign convention for all points)

    Rayleigh's quotient bounds the fundamental from ABOVE, so treat the result
    as an upper estimate of fn.
    """
    W = np.asarray(W, float)
    y = np.asarray(y, float)
    den = np.sum(W * y ** 2)
    if den <= 0:
        raise ValueError("deflections must be non-zero")
    return math.sqrt(G_IN * np.sum(W * y) / den) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# 2. Input environment
# ---------------------------------------------------------------------------
def _check_profile(f, a):
    f = np.asarray(f, float)
    a = np.asarray(a, float)
    if f.ndim != 1 or f.size < 2 or f.shape != a.shape:
        raise ValueError("f and levels must be 1-D arrays of equal length >= 2")
    if np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must be positive and strictly increasing")
    if np.any(a <= 0):
        raise ValueError("levels must be positive")
    return f, a


def grms(f, asd) -> float:
    """Overall Grms of a piecewise log-log ASD (Hz, g^2/Hz).

    Each segment is a straight line on log-log axes with slope
    m = log(a2/a1)/log(f2/f1); its area is integrated exactly.
    """
    f, a = _check_profile(f, asd)
    f1, f2, a1, a2 = f[:-1], f[1:], a[:-1], a[1:]
    r = f2 / f1
    m = np.log(a2 / a1) / np.log(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        area = np.where(np.isclose(m, -1.0),
                        a1 * f1 * np.log(r),
                        a1 * f1 / (m + 1.0) * (r ** (m + 1.0) - 1.0))
    return float(np.sqrt(area.sum()))


def asd_slope_point(a1: float, f1: float, f2: float, db_per_oct: float) -> float:
    """ASD at f2 given a1 at f1 and a slope in dB/octave (e.g. +6, -6)."""
    m = db_per_oct / (10.0 * math.log10(2.0))
    return a1 * (f2 / f1) ** m


def _loglog_interp(f, a, x):
    f, a = _check_profile(f, a)
    out = 10.0 ** np.interp(np.log10(x), np.log10(f), np.log10(a),
                            left=-np.inf, right=-np.inf)
    return float(out) if np.ndim(x) == 0 else out


def asd_at(f, asd, fn):
    """Input ASD at fn (g^2/Hz), log-log interpolated.  Zero outside the band."""
    return _loglog_interp(f, asd, fn)


# ---------------------------------------------------------------------------
# 3. Random response
# ---------------------------------------------------------------------------
def miles_3sigma(fn, Q, asd_fn, n_sigma: float = 3.0):
    """Miles' equation, n-sigma response acceleration (g).

        g_rms = sqrt( (pi/2) * fn * Q * ASD(fn) ),   g = n_sigma * g_rms

    Valid for one dominant mode with the input ASD roughly flat around fn.
    """
    return n_sigma * np.sqrt(np.pi / 2.0 * np.asarray(fn, float) * Q
                             * np.asarray(asd_fn, float))


def miles_envelope(f, asd, Q: float = 10.0, fn_range=None, npts: int = 400,
                   n_sigma: float = 3.0) -> tuple[float, float]:
    """Worst-case Miles response over an uncertain fn range.

    Returns (fn_worst, g_worst).  fn_range=None sweeps the whole profile band,
    which is very conservative; bound it with your fn estimate.
    """
    f, a = _check_profile(f, asd)
    lo, hi = fn_range if fn_range is not None else (f[0], f[-1])
    if not 0 < lo < hi:
        raise ValueError("fn_range must be (low, high) with 0 < low < high")
    fn = np.geomspace(lo, hi, npts)
    g = miles_3sigma(fn, Q, asd_at(f, a, fn), n_sigma)
    i = int(np.argmax(g))
    return float(fn[i]), float(g[i])


# ---------------------------------------------------------------------------
# 4. Sine response
# ---------------------------------------------------------------------------
def sine_response(fn: float, f, g_peak, Q: float = 10.0) -> float:
    """Peak sine response, g.

    fn inside the sine band : max(Q * A(fn), max A)   (resonant vs rigid body)
    fn outside the band     : max A                   (quasi-static)

    f, g_peak : sine profile breakpoints (Hz, g peak), log-log interpolated.
    """
    f, A = _check_profile(f, g_peak)
    rigid = float(A.max())
    if f[0] <= fn <= f[-1]:
        return max(Q * _loglog_interp(f, A, fn), rigid)
    return rigid


def sine_envelope(f, g_peak, Q: float = 10.0, fn_range=(0.0, 0.0),
                  npts: int = 400) -> tuple[float, float]:
    """Worst-case sine response over an uncertain fn range.

    Returns (fn_worst, g_worst).  Any overlap of fn_range with the sine band
    gives the resonant case.
    """
    lo, hi = fn_range
    if not 0 < lo < hi:
        raise ValueError("fn_range must be (low, high) with 0 < low < high")
    fn = np.geomspace(lo, hi, npts)
    g = np.array([sine_response(x, f, g_peak, Q) for x in fn])
    i = int(np.argmax(g))
    return float(fn[i]), float(g[i])


# ---------------------------------------------------------------------------
# 5. Limit acceleration per axis
# ---------------------------------------------------------------------------
def limit_g(random_3sigma: float = 0.0, sine_peak: float = 0.0,
            quasi_static: float = 0.0) -> float:
    """Limit g-level for one axis: the envelope of the separate environments.

    Random and sine are separate tests, so they are enveloped, not summed.
    If your spec combines random with a transient (e.g. RSS), do that first
    and pass the result as quasi_static.
    """
    return max(random_3sigma, sine_peak, quasi_static)


# ---------------------------------------------------------------------------
# 6. Interface forces and moments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InterfaceLoad:
    """Resultant load the component applies to the interface.

    Resolved at the bolt-pattern centroid.  z is normal to the interface,
    pointing away from the base, so +Fz pulls the joint apart.

    F : [Fx, Fy, Fz], lbf
    M : [Mx, My, Mz], lbf-in, about the pattern centroid
    """

    F: np.ndarray
    M: np.ndarray
    name: str = "case"


def interface_loads(W: float, g_vec, r_cg, name: str = "case") -> InterfaceLoad:
    """Rigid-body interface load: F = W * g, M = r_cg x F.

    W     : weight above the interface, lbf
    g_vec : [gx, gy, gz] limit g-levels
    r_cg  : CG position from the pattern centroid, in (z up from the faying
            surface)
    """
    F = W * np.asarray(g_vec, float)
    return InterfaceLoad(F, np.cross(np.asarray(r_cg, float), F), name)


def axis_interface_loads(W: float, r_cg, g_x: float, g_y: float,
                         g_z: float) -> list[InterfaceLoad]:
    """One interface load per test axis (random is applied one axis at a time)."""
    return [interface_loads(W, [g_x, 0, 0], r_cg, "X"),
            interface_loads(W, [0, g_y, 0], r_cg, "Y"),
            interface_loads(W, [0, 0, g_z], r_cg, "Z")]


# ---------------------------------------------------------------------------
# 7. Bolt load distribution
# ---------------------------------------------------------------------------
@dataclass
class BoltPattern:
    """Bolt locations in the interface plane, in (any origin)."""

    x: np.ndarray
    y: np.ndarray

    xc: float = field(init=False)
    yc: float = field(init=False)
    dx: np.ndarray = field(init=False)   # from centroid
    dy: np.ndarray = field(init=False)
    Ixx: float = field(init=False)       # sum dy^2, in^2
    Iyy: float = field(init=False)       # sum dx^2, in^2
    Ixy: float = field(init=False)       # sum dx*dy, in^2
    J: float = field(init=False)         # sum r^2, in^2

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.y = np.asarray(self.y, float)
        if self.x.shape != self.y.shape or self.x.ndim != 1 or self.x.size < 1:
            raise ValueError("x and y must be 1-D arrays of equal length")
        self.xc, self.yc = float(self.x.mean()), float(self.y.mean())
        self.dx, self.dy = self.x - self.xc, self.y - self.yc
        self.Ixx = float(np.sum(self.dy ** 2))
        self.Iyy = float(np.sum(self.dx ** 2))
        self.Ixy = float(np.sum(self.dx * self.dy))
        self.J = self.Ixx + self.Iyy

    @property
    def n(self) -> int:
        return int(self.x.size)

    @classmethod
    def circle(cls, n: int, radius: float, start_deg: float = 0.0) -> "BoltPattern":
        """n bolts equally spaced on a bolt circle of the given radius."""
        t = np.radians(start_deg) + 2.0 * np.pi * np.arange(n) / n
        return cls(radius * np.cos(t), radius * np.sin(t))

    @classmethod
    def rectangle(cls, nx: int, ny: int, sx: float, sy: float) -> "BoltPattern":
        """Perimeter of an nx-by-ny grid with pitches sx, sy (corners once)."""
        pts = {(i * sx, j * sy) for i in range(nx) for j in range(ny)
               if i in (0, nx - 1) or j in (0, ny - 1)}
        x, y = zip(*sorted(pts))
        return cls(np.array(x), np.array(y))


@dataclass(frozen=True)
class BoltLoads:
    """Per-bolt limit loads.  P > 0 is tension; V is the shear magnitude."""

    P: np.ndarray       # axial, lbf
    Vx: np.ndarray      # lbf
    Vy: np.ndarray      # lbf
    name: str = "case"

    @property
    def V(self) -> np.ndarray:
        return np.hypot(self.Vx, self.Vy)

    def critical(self) -> dict[str, tuple[int, float, float]]:
        """Bolts to carry into the joint checks: index, P_et, V.

        'max_tension' governs tension, separation and combined checks;
        'max_shear' governs shear, bearing and slip.  Check both.
        """
        iP = int(np.argmax(self.P))
        iV = int(np.argmax(self.V))
        P = np.maximum(self.P, 0.0)
        return {"max_tension": (iP, float(P[iP]), float(self.V[iP])),
                "max_shear": (iV, float(P[iV]), float(self.V[iV]))}


def bolt_loads(pattern: BoltPattern, load: InterfaceLoad,
               prying: float = 1.0) -> BoltLoads:
    """Distribute an interface load to the bolts (rigid flange, equal bolts).

    Axial: P_i = Fz/n + a*dy_i + b*dx_i, with a, b solved from moment
    equilibrium about the pattern centroid,

        [Ixx  Ixy] [a]   [ Mx]
        [Ixy  Iyy] [b] = [-My]

    which reduces to P_i = Fz/n + Mx*dy_i/Ixx - My*dx_i/Iyy when Ixy = 0.
    A preloaded joint that stays clamped rotates about the pattern centroid,
    so no heel pivot is used.  ``prying`` (>= 1) scales tensile bolt loads.

    Shear: direct share plus torsion,
        Vx_i = Fx/n - Mz*dy_i/J,   Vy_i = Fy/n + Mz*dx_i/J
    """
    if prying < 1.0:
        raise ValueError("prying factor must be >= 1")
    Fx, Fy, Fz = (float(v) for v in load.F)
    Mx, My, Mz = (float(v) for v in load.M)
    n, p = pattern.n, pattern

    A = np.array([[p.Ixx, p.Ixy], [p.Ixy, p.Iyy]])
    rhs = np.array([Mx, -My])
    ab = np.linalg.pinv(A) @ rhs
    if not np.allclose(A @ ab, rhs, rtol=1e-9, atol=1e-9 * max(1.0, np.abs(rhs).max())):
        raise ValueError("bolt pattern cannot react this moment (bolts collinear "
                         "or a single bolt); add bolts or model flange bearing")
    a, b = ab
    P = Fz / n + a * p.dy + b * p.dx
    P = np.where(P > 0, prying * P, P)

    if p.J > 0:
        Vx = Fx / n - Mz * p.dy / p.J
        Vy = Fy / n + Mz * p.dx / p.J
    elif abs(Mz) > 0:
        raise ValueError("a single bolt cannot react torsion Mz")
    else:
        Vx, Vy = np.full(n, Fx / n), np.full(n, Fy / n)
    return BoltLoads(P, np.asarray(Vx, float), np.asarray(Vy, float), load.name)


# ---------------------------------------------------------------------------
# 8. Joint checks
# ---------------------------------------------------------------------------
@dataclass
class VibeJointResult:
    """Merged margins: yield rows from the yield run, ultimate rows from the
    ultimate run, separation from the run carrying FF_sep * FS_sep."""

    results: list[M.MarginResult]
    yield_run: JointAnalysis
    ult_run: JointAnalysis

    @property
    def critical(self) -> M.MarginResult:
        return min(self.results, key=lambda r: r.MS)

    @property
    def passes(self) -> bool:
        return all(r.passes for r in self.results)

    def report(self) -> str:
        lines = [f"{'mode':<32}{'basis':<8}{'SF':>6}{'MS':>10}  status", "-" * 64]
        for r in sorted(self.results, key=lambda x: x.MS):
            ms = "inf" if r.MS == math.inf else f"{r.MS:.3f}"
            lines.append(f"{r.mode:<32}{r.basis:<8}{r.SF:>6.3f}{ms:>10}  "
                         f"{'PASS' if r.passes else 'FAIL'}")
        c = self.critical
        lines += ["-" * 64, f"Critical: {c.mode} ({c.basis}), MS = {c.MS:.3f}"]
        return "\n".join(lines)


def run_vibe_case(fastener, bolt_material, joint, preload, P_et: float, V: float,
                  *, FS_y: float = 1.25, FS_u: float = 1.4,
                  FF_y: float = 1.0, FF_u: float = 1.15,
                  FS_sep: float = M.SF_SEPARATION_STRUCTURAL, FF_sep: float = 1.0,
                  Mb: float = 0.0, delta_T: float = 0.0, name: str = "vibe",
                  **analysis_kwargs) -> VibeJointResult:
    """Run JointAnalysis with the correct factor on each strength basis.

    JointAnalysis applies one SF to both bases, so this runs it twice --
    SF = FF_y*FS_y for yield rows and FF_u*FS_u for ultimate rows -- and
    merges.  P_et and V are LIMIT loads per bolt (from bolt_loads).

    Defaults: FS 1.25 / 1.4 (NASA-STD-5001, test-verified metallic),
    FF_u = 1.15 (NASA-STD-5020 A.12), FF_y = 1.0 unless the joint is
    sensitive to local yielding, FF_sep = 1.0 unless separation-critical
    (then 1.15).  Set all of these per project.
    """
    runs = {}
    for basis, SF in (("yield", FF_y * FS_y), ("ult", FF_u * FS_u)):
        lc = LoadCase(name=f"{name} ({basis})", P_et=P_et, V=V, Mb=Mb,
                      delta_T=delta_T, SF_tension=SF, SF_shear=SF,
                      SF_separation=FF_sep * FS_sep)
        runs[basis] = JointAnalysis(fastener, bolt_material, joint, preload, lc,
                                    **analysis_kwargs)
    merged = [r for r in runs["yield"].results if r.basis == "yield"]
    merged += [r for r in runs["ult"].results if r.basis in ("ult", "n/a")]
    return VibeJointResult(merged, runs["yield"], runs["ult"])


def slip_margin(P_s: float, P_t: float, preload, mu: float, FS: float,
                FF: float = 1.0, P_thermal: float = 0.0) -> M.MarginResult:
    """Per-fastener joint slip.  NASA-STD-5020 eq. A.10-6.

        MS = mu * Pp_min / ( FF * FS * (P_s + mu * P_t) ) - 1

    P_s, P_t : limit shear and tension at that fastener, lbf
    mu       : 0.2 max for clean, uncoated, unlubricated metal; 0.1 for all
               other surfaces (coated, anodized, lubricated) unless tested
    FS       : factor for the failure mode slip would cause
    Uses single-bolt Po_min, which is conservative; NASA-STD-5020 A.2.1
    allows the pattern-mean minimum for slip.
    """
    Pp_min = preload.Po_min + min(P_thermal, 0.0)
    applied = FF * FS * (P_s + mu * max(P_t, 0.0))
    allow = mu * Pp_min
    return M.MarginResult("joint slip", "n/a", applied, allow, FF * FS,
                          M._ms(allow, applied), f"mu={mu}, Pp_min={Pp_min:.2f} lbf")


# ---------------------------------------------------------------------------
# 9. Fatigue
# ---------------------------------------------------------------------------
def bolt_alternating_stress(P_ext_1sigma: float, joint, fastener) -> float:
    """1-sigma alternating bolt stress, psi: n*phi*P / At.

    Valid only while the joint stays clamped (positive separation margin at
    the 3-sigma load).  Once separated the bolt sees the full external load.
    """
    return joint.n * joint.phi * P_ext_1sigma / fastener.At


def steinberg_cycles(fn: float, duration_s: float) -> np.ndarray:
    """Cycles at 1, 2 and 3 sigma: fn * T * (0.683, 0.271, 0.0433)."""
    return fn * duration_s * np.asarray(STEINBERG_FRACTIONS)


def steinberg_damage(S_1sigma: float, fn: float, duration_s: float, b: float,
                     S_ref: float, N_ref: float, life_factor: float = 4.0) -> float:
    """Miner damage under random vibration, Steinberg three-band method.

        n_k = fn * LF * T * {0.683, 0.271, 0.0433}
        N_k = N_ref * (S_ref / (k * S_1sigma))**b,   k = 1, 2, 3
        D   = sum(n_k / N_k)       must be < 1

    b, S_ref, N_ref : Basquin fit of an S-N curve for the part.  For bolts use
    data taken at a mean stress representative of preload.
    """
    if S_1sigma <= 0:
        return 0.0
    n = steinberg_cycles(fn, life_factor * duration_s)
    S = S_1sigma * np.array([1.0, 2.0, 3.0])
    N = N_ref * (S_ref / S) ** b
    return float(np.sum(n / N))


def low_risk_fatigue_screen(*, is_A286: bool, thread_form: str,
                            rolled_after_heat_treat: bool, P_tL: float,
                            P_tu_allow: float, fatigue_factor: float,
                            E_clamped_min: float, E_bolt: float,
                            edge_distance: float, d_major: float,
                            Pp_min: float) -> dict[str, bool]:
    """NASA-STD-5020 sec. 4.2 low-risk fatigue classification.

    Every entry must be True to skip fatigue analysis (Technical Authority
    approval still required).  thread_form: 'UNJ', 'MJ' or 'UNR'.
    """
    form = thread_form.upper()
    limit = {"UNJ": 0.50, "MJ": 0.50, "UNR": 0.40}.get(form)
    if limit is None:
        raise ValueError("thread_form must be 'UNJ', 'MJ' or 'UNR'")
    checks = {
        "A-286 fastener": is_A286,
        "threads rolled after heat treat": rolled_after_heat_treat,
        f"factored limit tension <= {limit:.0%} of ultimate":
            fatigue_factor * P_tL <= limit * P_tu_allow,
        "clamped modulus > E_bolt/3": E_clamped_min > E_bolt / 3.0,
        "edge distance >= 1.5 D": edge_distance >= 1.5 * d_major,
        "limit tension <= min preload": P_tL <= Pp_min,
    }
    checks["LOW RISK"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# 10. Test verification
# ---------------------------------------------------------------------------
def q_half_power(f, T) -> tuple[float, float]:
    """Q from a sine-survey transmissibility curve by the half-power method.

    Returns (fn, Q) with Q = fn / (f2 - f1), f1 and f2 where T = Tmax/sqrt(2).
    For light damping Q is also close to Tmax itself.
    """
    f = np.asarray(f, float)
    T = np.asarray(T, float)
    i = int(np.argmax(T))
    half = T[i] / math.sqrt(2.0)
    left = np.nonzero(T[:i] < half)[0]
    right = np.nonzero(T[i:] < half)[0]
    if left.size == 0 or right.size == 0:
        raise ValueError("survey does not cross the half-power level on both sides")
    j, k = left[-1], i + right[0]
    f1 = np.interp(half, [T[j], T[j + 1]], [f[j], f[j + 1]])
    f2 = np.interp(half, [T[k], T[k - 1]], [f[k], f[k - 1]])
    return float(f[i]), float(f[i] / (f2 - f1))


def signature_shift(fn_pre: float, fn_post: float,
                    tol: float = 0.05) -> tuple[float, bool]:
    """Fractional fn shift between pre- and post-test surveys, and pass flag.

    A drop beyond tol suggests slip, preload loss or damage.  The tolerance is
    set by the project test plan (5% is common).
    """
    shift = (fn_post - fn_pre) / fn_pre
    return shift, abs(shift) <= tol
