#!/usr/bin/env python3
"""
1000-candidate repeated preliminary benchmark
=============================================

Compare:
    1. reduced-space active-set HiGHS SQP
    2. direct full-space HiGHS SQP

Problem:
    S1000.txt -> Sigma
    A1000.txt -> Omega
    lambda = 0.5
    kappa = 1.0
    robust-gap tolerance = 1e-6

This script performs 5 paired runs and alternates the execution order
to reduce systematic timing bias.

Run from the RobustOCS repository root:

    python benchmark_1000_repeated5.py

Requires:
    benchmark_full_vs_active_v2.py

Outputs:
    benchmark_1000_repeated5_results.csv
    benchmark_1000_repeated5_summary.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import robustocs as rocs

import benchmark_full_vs_active_v2 as base


N_REPEATS = 5
ROBUST_GAP_TOL = 1e-6
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
            "failures": 0,
            "min_dim": None,
            "median_dim": None,
            "mean_dim": None,
            "max_dim": None,
            "full_dim_attempts": 0,
        }

    dims = np.asarray(
        [row["dimension"] for row in qp_records],
        dtype=float,
    )

    return {
        "attempts": len(qp_records),
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


def stat(values):
    values = [float(v) for v in values]
    return {
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    base.ROBUST_GAP_TOL = ROBUST_GAP_TOL
    base.LAM = LAM
    base.KAPPA = KAPPA
    base.REDUCED_COST_TOL = REDUCED_COST_TOL
    base.ACTIVE_TOL = ACTIVE_TOL

    problem = load_1000_problem()
    _, _, _, n, _, _, _ = problem

    if n != 1000:
        raise RuntimeError(f"Expected n=1000, but loaded n={n}.")

    print("=" * 92)
    print("1000-CANDIDATE REPEATED PRELIMINARY BENCHMARK")
    print("Reduced-space active-set HiGHS SQP vs direct full-space HiGHS SQP")
    print("=" * 92)
    print("Mapping: S1000 -> Sigma, A1000 -> Omega")
    print(f"lambda = {LAM}")
    print(f"kappa = {KAPPA}")
    print(f"robust-gap tolerance = {ROBUST_GAP_TOL:.1e}")
    print(f"repeats = {N_REPEATS}")
    print("Run order alternates between repeats.")
    print()

    rows = []

    for repeat in range(1, N_REPEATS + 1):
        with base.QPAttemptTracker() as tracker:
            if repeat % 2 == 1:
                reduced = base.run_instrumented_sqp(
                    base.qp_module.highs_active_set_qp,
                    problem,
                    "reduced-space",
                    repeat,
                    tracker,
                )
                full = base.run_instrumented_sqp(
                    base.full_space_highs_qp,
                    problem,
                    "full-space",
                    repeat,
                    tracker,
                )
            else:
                full = base.run_instrumented_sqp(
                    base.full_space_highs_qp,
                    problem,
                    "full-space",
                    repeat,
                    tracker,
                )
                reduced = base.run_instrumented_sqp(
                    base.qp_module.highs_active_set_qp,
                    problem,
                    "reduced-space",
                    repeat,
                    tracker,
                )

            attempt_records = tracker.records.copy()

        reduced_stats = qp_stats(attempt_records, "reduced-space")
        full_stats = qp_stats(attempt_records, "full-space")

        both_converged = bool(reduced["success"] and full["success"])

        speedup = None
        max_w_diff = None
        objective_diff = None
        z_diff = None

        if both_converged:
            speedup = full["runtime_s"] / reduced["runtime_s"]
            max_w_diff = float(
                np.max(np.abs(reduced["w"] - full["w"]))
            )
            objective_diff = abs(
                reduced["objective"] - full["objective"]
            )
            z_diff = abs(reduced["z"] - full["z"])

        row = {
            "repeat": repeat,
            "reduced_success": reduced["success"],
            "full_success": full["success"],
            "reduced_runtime_s": reduced["runtime_s"],
            "full_runtime_s": full["runtime_s"],
            "speedup_full_over_reduced": speedup,
            "reduced_sqp_subproblems": reduced["sqp_subproblems_attempted"],
            "full_sqp_subproblems": full["sqp_subproblems_attempted"],
            "reduced_robust_gap": reduced["robust_gap"],
            "full_robust_gap": full["robust_gap"],
            "reduced_objective": reduced["objective"],
            "full_objective": full["objective"],
            "max_abs_w_diff": max_w_diff,
            "objective_abs_diff": objective_diff,
            "z_abs_diff": z_diff,
            "reduced_nonzero_w": reduced["nonzero_w"],
            "full_nonzero_w": full["nonzero_w"],
            "reduced_qp_attempts": reduced_stats["attempts"],
            "reduced_qp_failures": reduced_stats["failures"],
            "reduced_qp_min_dim": reduced_stats["min_dim"],
            "reduced_qp_median_dim": reduced_stats["median_dim"],
            "reduced_qp_mean_dim": reduced_stats["mean_dim"],
            "reduced_qp_max_dim": reduced_stats["max_dim"],
            "reduced_full_dim_attempts": reduced_stats["full_dim_attempts"],
            "full_qp_attempts": full_stats["attempts"],
            "full_qp_failures": full_stats["failures"],
            "full_failed_sqp_iteration": full["failed_sqp_iteration"],
            "full_failed_rows": full["failed_rows"],
        }
        rows.append(row)

        print(f"Repeat {repeat}:")
        if reduced["success"]:
            print(
                f"  reduced-space: OK, {reduced['runtime_s']:.3f}s, "
                f"SQPs={reduced['sqp_subproblems_attempted']}, "
                f"gap={reduced['robust_gap']:.3e}"
            )
        else:
            print(
                f"  reduced-space: FAIL at SQP "
                f"{reduced['failed_sqp_iteration']}"
            )

        if full["success"]:
            print(
                f"  full-space:    OK, {full['runtime_s']:.3f}s, "
                f"SQPs={full['sqp_subproblems_attempted']}, "
                f"gap={full['robust_gap']:.3e}"
            )
        else:
            print(
                f"  full-space:    FAIL at SQP "
                f"{full['failed_sqp_iteration']}"
            )

        if speedup is not None:
            print(
                f"  speed-up full/reduced = {speedup:.3f}x"
            )
            print(
                f"  max|dw|={max_w_diff:.3e}, "
                f"|df|={objective_diff:.3e}"
            )
        else:
            print("  speed-up = N/A")

        print(
            f"  reduced QP dimension: median="
            f"{reduced_stats['median_dim']}, "
            f"mean={reduced_stats['mean_dim']:.3f}, "
            f"full-dim attempts={reduced_stats['full_dim_attempts']}"
        )
        print()

    csv_path = Path("benchmark_1000_repeated5_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    converged_pairs = [
        row for row in rows
        if row["reduced_success"] and row["full_success"]
    ]

    if converged_pairs:
        reduced_time_stats = stat(
            row["reduced_runtime_s"] for row in converged_pairs
        )
        full_time_stats = stat(
            row["full_runtime_s"] for row in converged_pairs
        )
        speedup_stats = stat(
            row["speedup_full_over_reduced"]
            for row in converged_pairs
        )
        reduced_qp_mean_stats = stat(
            row["reduced_qp_mean_dim"]
            for row in converged_pairs
        )
        reduced_qp_median_stats = stat(
            row["reduced_qp_median_dim"]
            for row in converged_pairs
        )

        max_solution_diff = max(
            row["max_abs_w_diff"] for row in converged_pairs
        )
        max_objective_diff = max(
            row["objective_abs_diff"] for row in converged_pairs
        )
    else:
        reduced_time_stats = None
        full_time_stats = None
        speedup_stats = None
        reduced_qp_mean_stats = None
        reduced_qp_median_stats = None
        max_solution_diff = None
        max_objective_diff = None

    reduced_successes = sum(row["reduced_success"] for row in rows)
    full_successes = sum(row["full_success"] for row in rows)

    lines = [
        "=" * 92,
        "1000-CANDIDATE REPEATED PRELIMINARY BENCHMARK SUMMARY",
        "=" * 92,
        "Mapping: S1000 -> Sigma, A1000 -> Omega",
        f"lambda = {LAM}",
        f"kappa = {KAPPA}",
        f"robust-gap tolerance = {ROBUST_GAP_TOL:.1e}",
        f"repeats = {N_REPEATS}",
        "",
        "Convergence",
        f"  reduced-space = {reduced_successes}/{N_REPEATS}",
        f"  full-space    = {full_successes}/{N_REPEATS}",
        f"  valid paired comparisons = {len(converged_pairs)}",
        "",
    ]

    if converged_pairs:
        lines.extend([
            "Reduced-space runtime (s)",
            f"  mean   = {reduced_time_stats['mean']:.6f}",
            f"  median = {reduced_time_stats['median']:.6f}",
            f"  stdev  = {reduced_time_stats['stdev']:.6f}",
            f"  min    = {reduced_time_stats['min']:.6f}",
            f"  max    = {reduced_time_stats['max']:.6f}",
            "",
            "Full-space runtime (s)",
            f"  mean   = {full_time_stats['mean']:.6f}",
            f"  median = {full_time_stats['median']:.6f}",
            f"  stdev  = {full_time_stats['stdev']:.6f}",
            f"  min    = {full_time_stats['min']:.6f}",
            f"  max    = {full_time_stats['max']:.6f}",
            "",
            "Speed-up = full-space / reduced-space",
            f"  mean   = {speedup_stats['mean']:.6f}x",
            f"  median = {speedup_stats['median']:.6f}x",
            f"  stdev  = {speedup_stats['stdev']:.6f}",
            f"  min    = {speedup_stats['min']:.6f}x",
            f"  max    = {speedup_stats['max']:.6f}x",
            "",
            "Solution agreement",
            f"  maximum max|w_reduced - w_full| = "
            f"{max_solution_diff:.12e}",
            f"  maximum objective absolute difference = "
            f"{max_objective_diff:.12e}",
            "",
            "Reduced-space QP dimensions across paired runs",
            f"  mean of per-run mean dimensions = "
            f"{reduced_qp_mean_stats['mean']:.3f}",
            f"  mean of per-run median dimensions = "
            f"{reduced_qp_median_stats['mean']:.3f}",
            f"  full QP dimension = {FULL_QP_DIMENSION}",
            "",
        ])
    else:
        lines.extend([
            "No valid runtime speed-up can be reported because there were",
            "no repeats in which both methods converged.",
            "",
        ])

    lines.extend([
        "Interpretation",
        "  This is a 5-run preliminary repeated experiment.",
        "  If convergence and speed-up are stable, use a 10-run version for",
        "  the final dissertation timing table.",
        "=" * 92,
    ])

    summary_text = "\n".join(lines)
    txt_path = Path("benchmark_1000_repeated5_summary.txt")
    txt_path.write_text(summary_text + "\n", encoding="utf-8")

    print(summary_text)
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
