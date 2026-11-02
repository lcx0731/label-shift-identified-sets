"""Deterministic synthetic examples and exhaustive finite-sample coverage."""

import argparse
from fractions import Fraction as F
import json
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import audit, multinomial_bands
from .reference import rational_vertices


FULL = np.array([[.7, .2, .1], [.2, .6, .2], [.1, .2, .7]])
SINGULAR = np.array([[.8, .8, .1], [.1, .1, .2], [.1, .1, .7]])
PRIOR = np.array([.2, .3, .5])


def exact_count_coverage(n=20, alpha=.05):
    """All three-bin count vectors; rational multinomial masses, numeric bands.

    These are exact summation benchmarks, not Monte Carlo estimates. Comparisons
    with beta quantiles and LP endpoints still use floating-point arithmetic.
    """
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 40:
        raise ValueError("enumeration requires integer n in [1, 40]")
    q = [F(1, 4), F(8, 25), F(43, 100)]  # FULL @ PRIOR exactly
    total = F(0)
    covered = F(0)
    rectangle_covered = F(0)
    empty = F(0)
    count = 0
    for i in range(n+1):
        for j in range(n-i+1):
            counts = [i, j, n-i-j]
            coefficient = factorial(n)
            for x in counts:
                coefficient //= factorial(x)
            mass = F(coefficient)
            for prob, x in zip(q, counts):
                mass *= prob**x
            total += mass
            bands = multinomial_bands(counts, alpha)
            count += 1
            if all(l <= float(prob) <= h for l, prob, h in
                   zip(bands["lower"], q, bands["upper"])):
                covered += mass
            report = audit(FULL, bands["lower"], bands["upper"])
            if report["status"] == "infeasible":
                empty += mass
            elif all(r["lower"]-1e-10 <= p <= r["upper"]+1e-10
                     for r, p in zip(report["ranges"], PRIOR)):
                rectangle_covered += mass
    assert total == 1
    return {"n": n, "alpha": alpha, "count_vectors": count,
            "probability_mass_sum": str(total),
            "joint_set_coverage": float(covered),
            "coordinate_rectangle_coverage": float(rectangle_covered),
            "empty_set_probability": float(empty),
            "joint_set_coverage_rational": str(covered)}


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    q = SINGULAR@PRIOR
    singular = audit(SINGULAR, q, q, [[1, 1, 0], [1, -1, 0]])
    vertices = rational_vertices(
        [[".8", ".8", ".1"], [".1", ".1", ".2"], [".1", ".1", ".7"]],
        [".45", ".15", ".40"], [".45", ".15", ".40"])
    assert vertices == [(F(0), F(1, 2), F(1, 2)),
                        (F(1, 2), F(0), F(1, 2))]
    full = audit(FULL, FULL@PRIOR, FULL@PRIOR)
    # Fixed count proportions: a deterministic width comparison, not random data.
    widths = []
    for n in [100, 1000, 10000, 100000]:
        for name, channel in [("full_rank", FULL), ("rank_deficient", SINGULAR)]:
            counts = np.rint(n*(channel@PRIOR)).astype(int)
            assert counts.sum() == n
            bands = multinomial_bands(counts)
            report = audit(channel, bands["lower"], bands["upper"])
            widths.append({"channel": name, "n": n, "counts": counts.tolist(),
                           "prior_0_lower": report["ranges"][0]["lower"],
                           "prior_0_upper": report["ranges"][0]["upper"],
                           "width": report["ranges"][0]["upper"]-
                                    report["ranges"][0]["lower"]})
    bad_q = np.array([.95, .03, .02])
    rejected = audit(FULL, bad_q, bad_q)
    # Observational equivalence: every new channel column equals FULL @ fake.
    actual = np.array([.8, .1, .1])
    fake = np.array([.1, .2, .7])
    observed = FULL@fake
    changed_channel = np.repeat(observed[:, None], 3, axis=1)
    np.testing.assert_allclose(changed_channel@actual, observed, atol=1e-15)
    accepted = audit(FULL, observed, observed)
    invisible = {"actual_prior": actual.tolist(), "apparent_prior": fake.tolist(),
                 "changed_channel": changed_channel.tolist(),
                 "observed_bin_probabilities": observed.tolist(),
                 "actual_to_apparent_l1_error": float(np.abs(actual-fake).sum()),
                 "audit_using_old_channel": accepted}
    summary = {"data": "Deterministic synthetic channels; no private or external data.",
               "true_prior": PRIOR.tolist(),
               "full_rank_channel": FULL.tolist(),
               "rank_deficient_channel": SINGULAR.tolist(),
               "full_rank_population": full,
               "rank_deficient_population": singular,
               "rational_rank_deficient_vertices": [[str(v) for v in p] for p in vertices],
               "width_experiment": widths,
               "incompatible_population": {"q": bad_q.tolist(), "audit": rejected},
               "invisible_conditional_shift": invisible,
               "enumerated_coverage": exact_count_coverage()}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    example = {"channel": SINGULAR.tolist(), "counts": [450, 150, 400],
               "alpha": .05, "contrasts": [[1, 1, 0], [1, -1, 0]]}
    (output/"example.json").write_text(json.dumps(example, indent=2)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    ranges = singular["ranges"]
    for i, r in enumerate(ranges):
        axs[0].plot([r["lower"], r["upper"]], [i, i], lw=5, color="#387888")
        axs[0].scatter([r["lower"], r["upper"]], [i, i], color="#387888", s=30)
    axs[0].set(yticks=range(5), yticklabels=["p₀", "p₁", "p₂", "p₀ + p₁", "p₀ − p₁"],
               xlabel="Population identified range", xlim=(-.6, .7),
               title="A singular channel still identifies a sum")
    axs[0].invert_yaxis()
    for name, color, label in [("full_rank", "#387888", "Full-rank channel"),
                               ("rank_deficient", "#c46c35", "Identical columns 0 and 1")]:
        rows = [r for r in widths if r["channel"] == name]
        axs[1].plot([r["n"] for r in rows], [r["width"] for r in rows],
                    marker="o", color=color, label=label)
    axs[1].axhline(.5, color="#c46c35", ls=":", alpha=.7)
    axs[1].set(xscale="log", xlabel="Target sample size (fixed proportions)",
               ylabel="Width for p₀", title="More data cannot resolve identical columns")
    axs[1].legend(fontsize=8)
    x = np.arange(3)
    axs[2].bar(x-.17, actual, width=.34, label="Actual under changed channel", color="#c46c35")
    axs[2].bar(x+.17, fake, width=.34, label="Apparent under old channel", color="#387888")
    axs[2].set(xticks=x, xticklabels=["Class 0", "Class 1", "Class 2"], ylim=(0, 1),
               ylabel="Class proportion", title="A zero residual can hide assumption failure")
    axs[2].legend(fontsize=8)
    fig.savefig(output/"identification.png", dpi=160, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(json.dumps({"output": str(output), "coverage": summary["enumerated_coverage"],
                      "incompatible_minimum_relaxation": rejected["minimum_band_relaxation"]},
                     indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    generate(parser.parse_args().output)


if __name__ == "__main__":
    main()
