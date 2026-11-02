import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import binomtest, multinomial

from shift_sets import multinomial_bands


@pytest.mark.parametrize("counts", [[0, 0, 20], [3, 8, 9], [20], [1, 1]])
def test_bands_match_independent_binomial_root_reference(counts):
    band = multinomial_bands(counts, .05)
    n = sum(counts)
    for i, count in enumerate(counts):
        ci = binomtest(count, n).proportion_ci(1-.05/len(counts), method="exact")
        assert band["lower"][i] == pytest.approx(ci.low, abs=2e-11)
        assert band["upper"][i] == pytest.approx(ci.high, abs=2e-11)


@pytest.mark.parametrize("q", [[.25, .32, .43], [.01, .09, .9], [0, .5, .5], [1, 0, 0]])
def test_exhaustive_multinomial_joint_band_coverage(q):
    n = 12
    coverage, total = 0., 0.
    for i in range(n+1):
        for j in range(n-i+1):
            counts = [i, j, n-i-j]
            mass = multinomial.pmf(counts, n, q)
            total += mass
            band = multinomial_bands(counts)
            if np.all(np.array(band["lower"]) <= q) and np.all(q <= np.array(band["upper"])):
                coverage += mass
    assert total == pytest.approx(1)
    assert coverage >= .95-1e-12


@pytest.mark.parametrize("counts", [[], [0, 0], [-1, 3], [1.5, 2], [True, False], [10**9+1]])
def test_bad_counts(counts):
    with pytest.raises(ValueError):
        multinomial_bands(counts)


@pytest.mark.parametrize("alpha", [0, 1, -.1, float('nan'), float('inf'), True, 1e-12])
def test_bad_alpha(alpha):
    with pytest.raises(ValueError):
        multinomial_bands([1, 2], alpha)


def run_cli(tmp_path, text):
    path = tmp_path/"input.json"
    path.write_text(text)
    before = path.read_bytes()
    result = subprocess.run([sys.executable, "-m", "shift_sets.cli", str(path)], capture_output=True, text=True)
    assert path.read_bytes() == before
    return result


def test_cli_counts(tmp_path):
    result = run_cli(tmp_path, json.dumps({"channel": [[.8, .2], [.2, .8]], "counts": [6, 4]}))
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert parsed["bands"]["method"] == "bonferroni-clopper-pearson"
    assert parsed["status"] == "feasible"


def test_cli_infeasible(tmp_path):
    result = run_cli(tmp_path, json.dumps({"channel": [[.8, .2], [.2, .8]],
                                         "lower": [.95, .05], "upper": [.95, .05]}))
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "infeasible"


@pytest.mark.parametrize("text", [
    '{"channel": [[1]], "channel": [[1]], "counts": [1]}',
    '{"channel": [[1]], "counts": [1], "lower": [1]}',
    '{"channel": [[1]], "lower": [1], "upper": [1], "alpha": 0.05}',
    '{"channel": [[1]], "counts": [1], "typo": 1}',
    '[]', '{bad json',
])
def test_cli_rejects_invalid_schema(tmp_path, text):
    result = run_cli(tmp_path, text)
    assert result.returncode == 1
    assert "input error" in result.stderr
