"""Accuracy and structural-quality benchmark (in the spirit of Fig. 5).

Measures the matching accuracy ``gamma`` (against the ground truth) and the
structural quality ``qS`` as a function of a structural perturbation applied to
otherwise isomorphic graphs, averaged over several random trials, and saves a
matplotlib figure.

Usage::

    python benchmark/accuracy_quality.py --mode quick
    python benchmark/accuracy_quality.py --mode full --platform GPU
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import gasm
from common import erdos_renyi, perturb_graph

# --- Editable parameters ------------------------------------------------
SCRIPT_PARAMETERS = {
    "quick": {"n": 30, "p": 0.15, "trials": 5, "edit_fractions": [0.0, 0.05, 0.1, 0.2]},
    "full": {"n": 100, "p": 0.1, "trials": 30, "edit_fractions": [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]},
}
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "accuracy_quality.png")
# ------------------------------------------------------------------------


def run(mode: str, platform: str, output: str) -> None:
    params = SCRIPT_PARAMETERS[mode]
    n, p, trials = params["n"], params["p"], params["trials"]
    fractions = params["edit_fractions"]

    n_edges_ref = max(int(p * n * (n - 1) / 2), 1)
    acc_mean, acc_std, qs_mean, qs_std = [], [], [], []

    for frac in fractions:
        n_edits = int(frac * n_edges_ref)
        accs, qss = [], []
        for t in range(trials):
            G = erdos_renyi(n, p, seed=t)
            H, gt = perturb_graph(G, n_edits, seed=1000 + t)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", gasm.utils.GASMWarning)
                M = gasm.match(G, H, platform=platform, seed=t)
            accs.append(M.accuracy(gt))
            qss.append(M.structural_quality(G, H))
        acc_mean.append(np.mean(accs))
        acc_std.append(np.std(accs))
        qs_mean.append(np.mean(qss))
        qs_std.append(np.std(qss))
        print(f"edit fraction {frac:.2f}: accuracy {acc_mean[-1]:.3f}  qS {qs_mean[-1]:.3f}")

    fractions = np.asarray(fractions)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(fractions, acc_mean, yerr=acc_std, marker="o", capsize=3, label="accuracy $\\gamma$")
    ax.errorbar(fractions, qs_mean, yerr=qs_std, marker="s", capsize=3, label="structural quality $q_S$")
    ax.set_xlabel("edge perturbation fraction")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"GASM accuracy and quality (n={n}, p={p}, {platform})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--platform", choices=["GPU", "CPU"], default="CPU")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.mode, args.platform, args.output)


if __name__ == "__main__":
    main()
