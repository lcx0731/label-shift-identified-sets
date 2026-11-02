"""Numerical LP bounds; statistical validity requires the stated channel model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta

LP_OPTIONS = {"primal_feasibility_tolerance": 1e-9,
              "dual_feasibility_tolerance": 1e-9}
CHECK_TOL = 2e-8


def _array(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D numeric array (no booleans)")
    a = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _inputs(channel, lower, upper):
    a = _array(channel, 2, "channel")
    if min(a.shape, default=0) < 1 or max(a.shape) > 100:
        raise ValueError("channel must have between 1 and 100 rows and columns")
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError("channel entries must be in [0, 1]")
    if not np.allclose(a.sum(axis=0), 1, rtol=0, atol=1e-12):
        raise ValueError("channel COLUMNS must sum to one; no normalization is applied")
    lo = _array(lower, 1, "lower")
    hi = _array(upper, 1, "upper")
    if lo.shape != (a.shape[0],) or hi.shape != lo.shape:
        raise ValueError("band lengths must equal the number of channel rows")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
        raise ValueError("require 0 <= lower <= upper <= 1")
    return a, lo, hi


def multinomial_bands(counts, alpha=0.05):
    """Bonferroni Clopper-Pearson bands for fixed-sample iid categorical counts.

    Each marginal has failure probability at most alpha / number_of_bins.
    Their dependence does not invalidate the union bound. Channel uncertainty
    and data-dependent choice of bins are NOT covered.
    """
    raw = np.asarray(counts)
    if raw.ndim != 1 or not 1 <= raw.size <= 100 or raw.dtype.kind not in "iu":
        raise ValueError("counts must be a nonempty integer vector with at most 100 bins")
    n = sum(int(k) for k in raw)
    if np.any(raw < 0) or not 1 <= n <= 1_000_000_000:
        raise ValueError("counts must be nonnegative with total in [1, 1000000000]")
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("alpha must be a finite number in [1e-10, 1)")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be numeric") from exc
    if not 1e-10 <= alpha < 1:
        raise ValueError("alpha must be in [1e-10, 1)")
    tail = alpha / (2 * len(raw))
    lo, hi = [], []
    for count in raw:
        k = int(count)
        lo.append(0.0 if k == 0 else float(beta.ppf(tail, k, n-k+1)))
        # isf avoids forming 1 - tail near one.
        hi.append(1.0 if k == n else float(beta.isf(tail, k+1, n-k)))
    if not np.all(np.isfinite(lo + hi)):
        raise ArithmeticError("nonfinite beta quantile")
    return {"lower": lo, "upper": hi, "alpha": alpha, "n": n,
            "method": "bonferroni-clopper-pearson"}


def _solve(c, g, h, eq, rhs, bounds):
    result = linprog(c, A_ub=g, b_ub=h, A_eq=eq, b_eq=rhs,
                     bounds=bounds, method="highs", options=LP_OPTIONS)
    if result.status not in (0, 2):
        raise ArithmeticError(f"LP did not resolve: {result.message}")
    return result


def _violation(p, a, lo, hi):
    return float(max(0.0, np.max(-p), abs(p.sum()-1),
                     np.max(lo-a@p), np.max(a@p-hi)))


def audit(channel, lower, upper, contrasts=None):
    """Bound each prior and optional linear contrasts over the SAME polytope.

    contrasts is an optional matrix whose rows are finite weight vectors.
    Endpoints include attaining priors, not a simultaneous rectangular model.
    Infeasibility is an LP outcome, not evidence of any particular cause.
    """
    a, lo, hi = _inputs(channel, lower, upper)
    b, k = a.shape
    if contrasts is None:
        custom = np.empty((0, k))
    else:
        custom = _array(contrasts, 2, "contrasts")
        if custom.shape[1] != k or custom.shape[0] > 100:
            raise ValueError("contrasts must have K columns and at most 100 rows")
    g, h = np.vstack([a, -a]), np.concatenate([hi, -lo])
    # Minimum common additive relaxation delta of every lower/upper band.
    relax = _solve(np.r_[np.zeros(k), 1.0],
                   np.column_stack([g, -np.ones(2*b)]), h,
                   np.array([np.r_[np.ones(k), 0.0]]), [1.0],
                   [(0, 1)]*k + [(0, None)])
    if not relax.success:
        raise ArithmeticError("relaxation LP unexpectedly infeasible")
    delta = float(relax.x[-1])
    relax_violation = _violation(relax.x[:-1], a, lo-delta, hi+delta)
    if relax_violation > CHECK_TOL:
        raise ArithmeticError("relaxation witness failed residual check")
    sv = np.linalg.svd(a, compute_uv=False)
    rank = int(np.linalg.matrix_rank(a))
    result = {"status": "infeasible", "n_bins": b, "n_classes": k,
              "channel_rank": rank, "singular_values": sv.tolist(),
              "condition_number": float(sv[0]/sv[-1]) if rank == k else None,
              "minimum_band_relaxation": delta,
              "relaxation_witness": relax.x[:-1].tolist(),
              "relaxation_witness_violation": relax_violation,
              "ranges": []}
    feasible = _solve(np.zeros(k), g, h, np.ones((1, k)), [1.0], [(0, 1)]*k)
    if not feasible.success:
        return result
    if _violation(feasible.x, a, lo, hi) > CHECK_TOL:
        raise ArithmeticError("feasible witness failed residual check")
    result.update(status="feasible", witness=feasible.x.tolist())
    for j, c in enumerate(np.vstack([np.eye(k), custom])):
        scale = float(np.max(np.abs(c))) or 1.0
        low = _solve(c/scale, g, h, np.ones((1, k)), [1.0], [(0, 1)]*k)
        high = _solve(-c/scale, g, h, np.ones((1, k)), [1.0], [(0, 1)]*k)
        if not low.success or not high.success:
            raise ArithmeticError("endpoint LP disagrees with feasibility LP")
        violation = max(_violation(low.x, a, lo, hi),
                        _violation(high.x, a, lo, hi))
        if violation > CHECK_TOL:
            raise ArithmeticError("endpoint witness failed residual check")
        low_value, high_value = float(c@low.x), float(c@high.x)
        if low_value > high_value + CHECK_TOL*scale:
            raise ArithmeticError("inverted endpoint range")
        result["ranges"].append({
            "name": f"prior[{j}]" if j < k else f"contrast[{j-k}]",
            "weights": c.tolist(), "lower": low_value, "upper": high_value,
            "lower_witness": low.x.tolist(), "upper_witness": high.x.tolist(),
            "max_constraint_violation": violation})
    return result
