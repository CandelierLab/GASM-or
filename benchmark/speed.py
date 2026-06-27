"""Processing-time benchmark (in the spirit of Fig. 10).

Measures the matching wall-clock time as a function of the graph size, for the
CPU and (when available) the GPU back-ends, and saves a log-log matplotlib
figure.

Usage::

    python benchmark/speed.py --mode quick
    python benchmark/speed.py --mode full --platforms CPU GPU
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

import gasm
from common import erdos_renyi, shuffle_graph

# --- Editable parameters ------------------------------------------------
SCRIPT_PARAMETERS = {
    "quick": {"sizes": [20, 50, 100, 200], "p": 0.1, "reps": 3},
    "full": {"sizes": [50, 100, 200, 500, 1000, 2000], "p": 0.05, "reps": 5},
}
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "speed.png")
# ------------------------------------------------------------------------


def _time_match(G, H, platform, reps):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gasm.utils.GASMWarning)
            gasm.match(G, H, platform=platform, seed=0)
        best = min(best, time.perf_counter() - t0)
    return best


def run(mode: str, platforms: list[str], output: str) -> None:
    params = SCRIPT_PARAMETERS[mode]
    sizes, p, reps = params["sizes"], params["p"], params["reps"]

    results = {plat: [] for plat in platforms}
    for n in sizes:
        G = erdos_renyi(n, p, seed=0)
        H, _ = shuffle_graph(G, seed=1)
        for plat in platforms:
            dt = _time_match(G, H, plat, reps)
            results[plat].append(dt)
            print(f"n={n:5d}  {plat}: {dt * 1e3:.1f} ms")

    fig, ax = plt.subplots(figsize=(6, 4))
    for plat in platforms:
        ax.loglog(sizes, results[plat], marker="o", label=plat)
    ax.set_xlabel("number of vertices $n$")
    ax.set_ylabel("matching time (s)")
    ax.set_title(f"GASM processing time (p={p})")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--platforms", nargs="+", choices=["GPU", "CPU"], default=["CPU"])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.mode, args.platforms, args.output)


if __name__ == "__main__":
    main()
