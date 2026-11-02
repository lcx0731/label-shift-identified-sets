from fractions import Fraction as F

import numpy as np
import pytest

from shift_sets import audit
from shift_sets.reference import rational_vertices


def test_identity_channel_recovers_population():
    p = [.1, .7, .2]
    r = audit(np.eye(3), p, p)
    assert r["status"] == "feasible"
    np.testing.assert_allclose([x["lower"] for x in r["ranges"]], p)
    np.testing.assert_allclose([x["upper"] for x in r["ranges"]], p)


def test_uninformative_channel_keeps_entire_simplex():
    r = audit([[.4]*3, [.6]*3], [.4, .6], [.4, .6], [[2, -3, 7]])
    assert r["channel_rank"] == 1
    assert r["condition_number"] is None
    np.testing.assert_allclose([x["lower"] for x in r["ranges"]], [0, 0, 0, -3])
    np.testing.assert_allclose([x["upper"] for x in r["ranges"]], [1, 1, 1, 7])


def test_partial_identification_and_attaining_witnesses():
    a = np.array([[.8, .8, .1], [.1, .1, .2], [.1, .1, .7]])
    q = np.array([.45, .15, .4])
    result = audit(a, q, q, [[1, 1, 0], [1, -1, 0]])
    np.testing.assert_allclose([r["lower"] for r in result["ranges"]], [0, 0, .5, .5, -.5], atol=1e-10)
    np.testing.assert_allclose([r["upper"] for r in result["ranges"]], [.5, .5, .5, .5, .5], atol=1e-10)
    for r in result["ranges"]:
        for end in ["lower", "upper"]:
            p = np.array(r[end+"_witness"])
            np.testing.assert_allclose(a@p, q, atol=1e-10)
            assert sum(p) == pytest.approx(1)
            assert p.min() >= -1e-10
            assert np.array(r["weights"])@p == pytest.approx(r[end])


@pytest.mark.parametrize("seed", range(12))
def test_independent_rational_active_set_oracle(seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 10, size=(3, 3))
    a = [[F(int(counts[i, j]), int(counts[:, j].sum())) for j in range(3)] for i in range(3)]
    prior = [F(1, 5), F(3, 10), F(1, 2)]
    q = [sum(x*p for x, p in zip(row, prior)) for row in a]
    lo, hi = [max(F(0), x-F(1, 20)) for x in q], [min(F(1), x+F(1, 20)) for x in q]
    vertices = rational_vertices(a, lo, hi)
    assert vertices
    c = [F(2), F(-1), F(3, 10)]
    result = audit(np.array(a, float), np.array(lo, float), np.array(hi, float), [list(map(float, c))])
    for r in result["ranges"]:
        weights = c if r["name"].startswith("contrast") else [F(int(j == int(r["name"][6]))) for j in range(3)]
        values = [sum(x*y for x, y in zip(weights, p)) for p in vertices]
        assert r["lower"] == pytest.approx(float(min(values)), abs=1e-9)
        assert r["upper"] == pytest.approx(float(max(values)), abs=1e-9)


def test_infeasibility_and_known_minimum_relaxation():
    r = audit([[.8, .2], [.2, .8]], [.95, .05], [.95, .05])
    assert r["status"] == "infeasible"
    assert not r["ranges"]
    assert r["minimum_band_relaxation"] == pytest.approx(.15)
    np.testing.assert_allclose(r["relaxation_witness"], [1, 0])
    assert not rational_vertices([[F(4, 5), F(1, 5)], [F(1, 5), F(4, 5)]],
                                 [F(19, 20), F(1, 20)], [F(19, 20), F(1, 20)])


def test_bands_need_not_contain_any_probability_vector():
    r = audit(np.eye(2), [.7, .7], [.8, .8])
    assert r["status"] == "infeasible"
    assert r["minimum_band_relaxation"] == pytest.approx(.2)


def test_band_widening_cannot_shrink_ranges():
    a = np.array([[.7, .3], [.3, .7]])
    narrow = audit(a, [.48, .48], [.52, .52])
    wide = audit(a, [.4, .4], [.6, .6])
    for n, w in zip(narrow["ranges"], wide["ranges"]):
        assert w["lower"] <= n["lower"] and w["upper"] >= n["upper"]


@pytest.mark.parametrize("scale", [0., 1e-14, -1., 1e12])
def test_linear_contrast_scaling(scale):
    r = audit([[1., 1.]], [1.], [1.], [[scale, -scale]])["ranges"][-1]
    assert r["lower"] == pytest.approx(-abs(scale), abs=1e-25)
    assert r["upper"] == pytest.approx(abs(scale), abs=1e-25)


def test_one_class_and_zero_probability_bin():
    r = audit([[1.], [0.]], [1., 0.], [1., 0.])
    assert r["ranges"][0]["lower"] == r["ranges"][0]["upper"] == 1.
    assert rational_vertices([[1], [0]], [1, 0], [1, 0]) == [(F(1),)]


def test_simplex_boundary_identifies_prior_despite_rank_deficiency():
    r = audit([[1., 0., 0.], [0., 1., 1.]], [1., 0.], [1., 0.])
    assert r["channel_rank"] == 2
    np.testing.assert_allclose([x["lower"] for x in r["ranges"]], [1, 0, 0])
    np.testing.assert_allclose([x["upper"] for x in r["ranges"]], [1, 0, 0])


def test_invisible_conditional_shift_cannot_be_diagnosed():
    a = np.array([[.7, .2, .1], [.2, .6, .2], [.1, .2, .7]])
    actual, apparent = np.array([.8, .1, .1]), np.array([.1, .2, .7])
    q = a@apparent
    new_channel = np.repeat(q[:, None], 3, axis=1)
    np.testing.assert_allclose(new_channel@actual, q)
    r = audit(a, q, q)
    assert r["status"] == "feasible"
    np.testing.assert_allclose([x["lower"] for x in r["ranges"]], apparent)
    assert not np.allclose(apparent, actual)


def test_permuting_bins_and_classes_preserves_answers():
    a = np.array([[.7, .2, .1], [.2, .6, .2], [.1, .2, .7]])
    lo, hi = np.array([.2, .25, .35]), np.array([.3, .4, .5])
    original = audit(a, lo, hi)
    rows, cols = [2, 0, 1], [1, 2, 0]
    changed = audit(a[rows][:, cols], lo[rows], hi[rows])
    for j, c in enumerate(cols):
        for end in ["lower", "upper"]:
            assert changed["ranges"][j][end] == pytest.approx(original["ranges"][c][end])


@pytest.mark.parametrize("a,lo,hi", [
    ([[.2, .4], [.5, .6]], [0, 0], [1, 1]),
    ([[1.1], [-.1]], [0, 0], [1, 1]),
    ([[float('nan')]], [0], [1]),
    ([[1.]], [.8], [.7]),
    ([[1.]], [-.1], [1]),
    ([[1.]], [0], [1.1]),
    ([[1.]], [0, 0], [1, 1]),
    ([], [], []), ([[True]], [0], [1]),
])
def test_invalid_inputs(a, lo, hi):
    with pytest.raises(ValueError):
        audit(a, lo, hi)


def test_invalid_contrast():
    with pytest.raises(ValueError):
        audit(np.eye(2), [0, 0], [1, 1], [[1, 2, 3]])
