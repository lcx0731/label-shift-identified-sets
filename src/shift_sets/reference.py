"""Independent small rational vertex oracle. No floating point or LP solver."""

from fractions import Fraction as F
from itertools import combinations
from math import comb


def _solve_square(rows, rhs):
    n = len(rhs)
    m = [list(row)+[value] for row, value in zip(rows, rhs)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if m[i][j]), None)
        if pivot is None:
            return None
        m[j], m[pivot] = m[pivot], m[j]
        value = m[j][j]
        m[j] = [x/value for x in m[j]]
        for i in range(n):
            if i != j:
                factor = m[i][j]
                m[i] = [x-factor*y for x, y in zip(m[i], m[j])]
    return tuple(row[-1] for row in m)


def rational_vertices(channel, lower, upper):
    """Enumerate vertices of simplex intersected with rational linear bands.

    Pass ints, Fractions, or decimal strings to specify exact intended values.
    A float is interpreted as its exact binary rational. Small test oracle only.
    """
    a = [[F(x) for x in row] for row in channel]
    lo, hi = list(map(F, lower)), list(map(F, upper))
    if not a or not a[0] or len(lo) != len(a) or len(hi) != len(a):
        raise ValueError("inconsistent dimensions")
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("ragged channel")
    constraints = [(row, h) for row, h in zip(a, hi)]
    constraints += [([-x for x in row], -l) for row, l in zip(a, lo)]
    constraints += [([F(-int(i == j)) for i in range(k)], F(0)) for j in range(k)]
    if comb(len(constraints), k-1) > 100_000:
        raise ValueError("reference limited to 100000 active-set combinations")
    found = set()
    for active in combinations(constraints, k-1):
        rows = [[F(1)]*k] + [row for row, _ in active]
        rhs = [F(1)] + [value for _, value in active]
        p = _solve_square(rows, rhs)
        if p is not None and all(sum(x*y for x, y in zip(row, p)) <= h
                                 for row, h in constraints):
            found.add(p)
    return sorted(found)
