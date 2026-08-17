#!/usr/bin/env python3
"""Validate Ch5 grid HPD intervals vs continuous MAP + Laplace approximation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ch5_core as c
from ch5_datasets import CH5_DATASET_KEYS, ch5_unpack_dataset


def _load_nll_fn():
    nb = Path("logistic-regression-chap4.ipynb")
    src = "".join(json.loads(nb.read_text())["cells"][1]["source"])
    g: dict = {}
    exec(compile(src, str(nb), "exec"), g)
    return g["_ch3_nll_sum_on_flat_grid"]


def _fmt_interval(iv):
    lo, hi = iv
    return f"[{lo:+.3f}, {hi:+.3f}]"


def main() -> int:
    nll_fn = _load_nll_fn()
    print("Ch5 credible-region validation (Gaussian prior, σ={})".format(c.CH5_PRIOR_SIGMA))
    print("=" * 72)
    worst = 0.0
    for key in CH5_DATASET_KEYS:
        study, exam, y = ch5_unpack_dataset(key)
        grid = c.ch5_posterior_3d_pack(study, exam, y, nll_fn=nll_fn, prior_kind="gaussian")
        cont = c.ch5_posterior_map_continuous(study, exam, y, nll_fn=nll_fn, prior_kind="gaussian")
        lap, _ = c.ch5_laplace_marginal_intervals(
            study, exam, y,
            (cont["ws"], cont["we"], cont["bb"]),
            nll_fn=nll_fn, prior_kind="gaussian",
        )
        print(f"\n{key}  covered mass={grid['covered']:.3f} (target {grid['mass']})")
        print(
            f"  grid MAP ({grid['ws']:+.3f}, {grid['we']:+.3f}, {grid['bb']:+.3f})"
            f"  |  continuous MAP ({cont['ws']:+.3f}, {cont['we']:+.3f}, {cont['bb']:+.3f})"
        )
        for axis in ("st", "el", "b"):
            giv = grid["intervals"][axis]
            liv = lap[axis]
            # Compare overlap: width ratio and center shift vs Laplace
            gw = giv[1] - giv[0]
            lw = liv[1] - liv[0]
            gc = 0.5 * (giv[0] + giv[1])
            lc = 0.5 * (liv[0] + liv[1])
            shift = abs(gc - lc)
            ratio = gw / max(lw, 1e-9)
            worst = max(worst, shift, abs(ratio - 1.0))
            print(
                f"  {axis}: HPD grid {_fmt_interval(giv)}"
                f"  |  Laplace {_fmt_interval(liv)}"
                f"  (width ratio {ratio:.2f}, center Δ {shift:.3f})"
            )
    print("\n" + "=" * 72)
    if worst > 0.8:
        print(f"WARN: largest discrepancy ~{worst:.2f} — check grid resolution or prior σ.")
        return 1
    print(f"OK: grid HPD marginals reasonably track Laplace (max discrepancy ~{worst:.2f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
