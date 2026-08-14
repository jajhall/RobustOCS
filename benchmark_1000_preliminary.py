#!/usr/bin/env python3
"""
Preliminary 1000-candidate Robust OCS benchmark
===============================================

Compares, once:

    1. reduced-space active-set HiGHS SQP
    2. direct full-space HiGHS SQP

using the dissertation mapping:

    S1000.txt -> Sigma
    A1000.txt -> Omega

and robust-gap tolerance 1e-6 by default.

This is a PRELIMINARY convergence/comparison run, not yet the final repeated
timing experiment.

Requirements
------------
Place this file in the RobustOCS repository root together with:

    benchmark_full_vs_active_v2.py

Run:

    python benchmark_1000_preliminary.py

Optional:

    python benchmark_1000_preliminary.py --tol 1e-6

Outputs
-------
    benchmark_1000_preliminary_results.csv
    benchmark_1000_preliminary_summary.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import robustocs as rocs
import benchmark_full_vs_active_v2 as base


LAM = 0.5
KAPPA = 1.0
REDUCED_COST_TOL = 1e-8
ACTIVE_TOL = 1e-10
FULL_QP_DIMENSION = 1001


def load_1000_problem():
    data_dir = Path("examples") / "1000"

    required = [
        data_dir / "S1000.txt",
        data_dir / "A1000.txt",
        data_dir / "EBV1000.txt",
        data_dir / "SEX1000.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run this script from the RobustOCS repository root.\n"
            "Missing files:\n" + "\n".join(missing)
        )

    # Dissertation mapping explicitly fixed by the project:
    # S1000 -> Sigma, A1000 -> Omega
    return rocs.load_problem(
        sigma_filename=str(data_dir / "S1000.txt"),
        mu_filename=str(data_dir / "EBV1000.txt"),
        omega_filename=str(data_dir / "A1000.txt"),
        sex_filename=str(data_dir / "SEX1000.txt"),
        issparse=True,
    )


def qp_stats(records, method):
    qp_records = [
        row for row in records
        if row["method"] == method and row["problem_type"] == "QP"
    ]

    if not qp_records:
        return {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "min_dim": None,
            "median_dim": None,
            "mean_dim": None,
            "max_dim": None,
            "full_dim_attempts": 0,
        }

    dims = np.asarray([row["dimension"] for row in qp_records], dtype=float)

    return {
        "attempts": len(qp_records),
        "successes": sum(bool(row["success"]) for row in qp_records),
        "failures": sum(not bool(row["success"]) for row in qp_records),
        "min_dim": int(np.min(dims)),
        "median_dim": float(np.median(dims)),
        "mean_dim": float(np.mean(dims)),
        "max_dim": int(np.max(dims)),
        "full_dim_attempts": sum(
            int(row["dimension"]) == FULL_QP_DIMENSION
            for row in qp_records
        ),
    }


def describe_result(result):
    if result["success"]:
        return (
            f"CONVERGED; runtime={result['runtime_s']:.6f}s; "
            f"SQP subproblems={result['sqp_subproblems_attempted']}; "
            f"robust gap={result['robust_gap']:.3e}; "
            f"objective={result['objective']:.12e}; "
            f"nonzero w={result['nonzero_w']}"
        )

    return (
        f"FAILED; runtime to failure={result['runtime_s']:.6f}s; "
        f"failed SQP iteration={result['failed_sqp_iteration']}; "
        f"planes={result['failed_planes']}; "
        f"rows={result['failed_rows']}; "
        f"last successful gap={result['last_success_gap']}; "
        f"failure={result['failure_type']}: {result['failure_message']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Outer SQP robust-gap tolerance (default: 1e-6).",
    )
    args = parser.parse_args()

    robust_tol = float(args.tol)

    # Keep the numerical settings aligned with the current dissertation tests.
    base.ROBUST_GAP_TOL = robust_tol
    base.LAM = LAM
    base.KAPPA = KAPPA
    base.REDUCED_COST_TOL = REDUCED_COST_TOL
    base.ACTIVE_TOL = ACTIVE_TOL

    problem = load_1000_problem()
    _, _, _, n, _, _, _ = problem

    if n != 1000:
        raise RuntimeError(f"Expected n=1000, but loaded n={n}.")

    print("=" * 92)
    print("1000-CANDIDATE PRELIMINARY BENCHMARK")
    print("Reduced-space active-set HiGHS SQP vs direct full-space HiGHS SQP")
    print("=" * 92)
    print("Mapping: S1000 -> Sigma, A1000 -> Omega")
    print(f"lambda = {LAM}")
    print(f"kappa = {KAPPA}")
    print(f"robust-gap tolerance = {robust_tol:.1e}")
    print(f"reduced-cost tolerance = {REDUCED_COST_TOL:.1e}")
    print(f"active tolerance = {ACTIVE_TOL:.1e}")
    print("No warm-up is used: this is a preliminary one-pair run.")
    print()

    with base.QPAttemptTracker() as tracker:
        print("Running reduced-space method...")
        reduced = base.run_instrumented_sqp(
            base.qp_module.highs_active_set_qp,
            problem,
            "reduced-space",
            1,
            tracker,
        )
        print("  " + describe_result(reduced))
        print()

        print("Running direct full-space method...")
        full = base.run_instrumented_sqp(
            base.full_space_highs_qp,
            problem,
            "full-space",
            1,
            tracker,
        )
        print("  " + describe_result(full))
        print()

        qp_attempt_records = tracker.records.copy()

    reduced_qp = qp_stats(qp_attempt_records, "reduced-space")
    full_qp = qp_stats(qp_attempt_records, "full-space")

    both_converged = bool(reduced["success"] and full["success"])

    speedup = None
    max_w_diff = None
    objective_diff = None
    z_diff = None

    if both_converged:
        speedup = full["runtime_s"] / reduced["runtime_s"]
        max_w_diff = float(np.max(np.abs(reduced["w"] - full["w"])))
        objective_diff = abs(reduced["objective"] - full["objective"])
        z_diff = abs(reduced["z"] - full["z"])

    print("=" * 92)
    print("COMPARISON")
    print("=" * 92)
    print(f"reduced-space convergence = {reduced['success']}")
    print(f"full-space convergence    = {full['success']}")

    if both_converged:
        print(f"preliminary speed-up full/reduced = {speedup:.6f}x")
        print(f"max |w_reduced - w_full|          = {max_w_diff:.12e}")
        print(f"|objective_reduced - objective_full| = {objective_diff:.12e}")
        print(f"|z_reduced - z_full|              = {z_diff:.12e}")
    else:
        print(
            "speed-up = N/A because both methods must converge before their "
            "solution times can be compared."
        )

    print()
    print("Reduced-space QP dimensions passed to HiGHS:")
    print(f"  QP attempts              = {reduced_qp['attempts']}")
    print(f"  failed QP attempts       = {reduced_qp['failures']}")
    if reduced_qp["attempts"]:
        print(f"  minimum dimension        = {reduced_qp['min_dim']}")
        print(f"  median dimension         = {reduced_qp['median_dim']:.1f}")
        print(f"  mean dimension           = {reduced_qp['mean_dim']:.3f}")
        print(f"  maximum dimension        = {reduced_qp['max_dim']}")
        print(
            f"  full-dimension attempts  = "
            f"{reduced_qp['full_dim_attempts']}"
        )

    print()
    print("Direct full-space QP dimensions passed to HiGHS:")
    print(f"  QP attempts              = {full_qp['attempts']}")
    print(f"  failed QP attempts       = {full_qp['failures']}")
    if full_qp["attempts"]:
        print(f"  minimum dimension        = {full_qp['min_dim']}")
        print(f"  median dimension         = {full_qp['median_dim']:.1f}")
        print(f"  mean dimension           = {full_qp['mean_dim']:.3f}")
        print(f"  maximum dimension        = {full_qp['max_dim']}")

    # ------------------------------------------------------------------
    # Save one-row machine-readable result
    # ------------------------------------------------------------------
    result_row = {
        "tolerance": robust_tol,
        "reduced_success": reduced["success"],
        "full_success": full["success"],
        "reduced_runtime_s": reduced["runtime_s"],
        "full_runtime_s": full["runtime_s"],
        "reduced_sqp_subproblems": reduced["sqp_subproblems_attempted"],
        "full_sqp_subproblems": full["sqp_subproblems_attempted"],
        "reduced_final_or_last_gap": reduced["robust_gap"],
        "full_final_or_last_gap": full["robust_gap"],
        "reduced_objective": reduced["objective"],
        "full_objective": full["objective"],
        "reduced_nonzero_w": reduced["nonzero_w"],
        "full_nonzero_w": full["nonzero_w"],
        "full_failed_sqp_iteration": full["failed_sqp_iteration"],
        "full_failed_planes": full["failed_planes"],
        "full_failed_rows": full["failed_rows"],
        "preliminary_speedup_full_over_reduced": speedup,
        "max_abs_w_diff": max_w_diff,
        "objective_abs_diff": objective_diff,
        "z_abs_diff": z_diff,
        "reduced_qp_attempts": reduced_qp["attempts"],
        "reduced_qp_failures": reduced_qp["failures"],
        "reduced_qp_min_dimension": reduced_qp["min_dim"],
        "reduced_qp_median_dimension": reduced_qp["median_dim"],
        "reduced_qp_mean_dimension": reduced_qp["mean_dim"],
        "reduced_qp_max_dimension": reduced_qp["max_dim"],
        "reduced_full_dimension_attempts": reduced_qp["full_dim_attempts"],
        "full_qp_attempts": full_qp["attempts"],
        "full_qp_failures": full_qp["failures"],
    }

    csv_path = Path("benchmark_1000_preliminary_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_row.keys()))
        writer.writeheader()
        writer.writerow(result_row)

    lines = [
        "=" * 92,
        "1000-CANDIDATE PRELIMINARY BENCHMARK SUMMARY",
        "=" * 92,
        "Mapping: S1000 -> Sigma, A1000 -> Omega",
        f"lambda = {LAM}",
        f"kappa = {KAPPA}",
        f"robust-gap tolerance = {robust_tol:.1e}",
        "",
        "Reduced-space result",
        "  " + describe_result(reduced),
        "",
        "Full-space result",
        "  " + describe_result(full),
        "",
        "Comparison",
        f"  both converged = {both_converged}",
        (
            f"  preliminary speed-up full/reduced = {speedup:.6f}x"
            if speedup is not None
            else "  preliminary speed-up = N/A"
        ),
        (
            f"  max |w_reduced - w_full| = {max_w_diff:.12e}"
            if max_w_diff is not None
            else "  max solution difference = N/A"
        ),
        (
            f"  objective absolute difference = {objective_diff:.12e}"
            if objective_diff is not None
            else "  objective absolute difference = N/A"
        ),
        "",
        "Reduced-space QP dimension statistics",
        f"  attempts = {reduced_qp['attempts']}",
        f"  failures = {reduced_qp['failures']}",
        f"  min = {reduced_qp['min_dim']}",
        f"  median = {reduced_qp['median_dim']}",
        f"  mean = {reduced_qp['mean_dim']}",
        f"  max = {reduced_qp['max_dim']}",
        f"  full-dimension attempts = {reduced_qp['full_dim_attempts']}",
        "",
        "Full-space QP dimension statistics",
        f"  attempts = {full_qp['attempts']}",
        f"  failures = {full_qp['failures']}",
        f"  min = {full_qp['min_dim']}",
        f"  median = {full_qp['median_dim']}",
        f"  mean = {full_qp['mean_dim']}",
        f"  max = {full_qp['max_dim']}",
        "",
        "Interpretation",
        "  This is a one-pair preliminary run. Runtime is not yet the final",
        "  dissertation benchmark. If both methods converge, repeat the",
        "  experiment multiple times before reporting mean runtime or speed-up.",
        "=" * 92,
    ]

    summary_text = "\n".join(lines)
    txt_path = Path("benchmark_1000_preliminary_summary.txt")
    txt_path.write_text(summary_text + "\n", encoding="utf-8")

    print()
    print(summary_text)
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
