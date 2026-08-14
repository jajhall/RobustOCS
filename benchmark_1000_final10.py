#!/usr/bin/env python3
"""
FINAL 1000-candidate repeated benchmark
=======================================

Compares:
    1. reduced-space active-set HiGHS SQP
    2. direct full-space HiGHS SQP

Problem definition used in the dissertation:
    S1000.txt -> Sigma
    A1000.txt -> Omega
    lambda = 0.5
    kappa = 1.0
    robust-gap tolerance = 1e-6

Design:
    - one unreported warm-up pair
    - 10 timed paired repeats
    - execution order alternates between repeats
    - speed-up is reported only when both methods converge

Run from the RobustOCS repository root:

    python benchmark_1000_final10.py

Requires:
    benchmark_full_vs_active_v2.py

Outputs:
    benchmark_1000_final10_results.csv
    benchmark_1000_final10_summary.csv
    benchmark_1000_final10_summary.txt
"""

from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import robustocs as rocs

import benchmark_full_vs_active_v2 as base


N_REPEATS = 10
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

    # Dissertation mapping:
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

    dims = np.asarray(
        [row["dimension"] for row in qp_records],
        dtype=float,
    )

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


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }

    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def fmt(value, digits=6):
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def run_pair(problem, repeat):
    with base.QPAttemptTracker() as tracker:
        # Alternate order to reduce systematic timing bias.
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

        records = tracker.records.copy()

    reduced_qp = qp_stats(records, "reduced-space")
    full_qp = qp_stats(records, "full-space")

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

    return {
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
        "reduced_qp_attempts": reduced_qp["attempts"],
        "reduced_qp_successes": reduced_qp["successes"],
        "reduced_qp_failures": reduced_qp["failures"],
        "reduced_qp_min_dim": reduced_qp["min_dim"],
        "reduced_qp_median_dim": reduced_qp["median_dim"],
        "reduced_qp_mean_dim": reduced_qp["mean_dim"],
        "reduced_qp_max_dim": reduced_qp["max_dim"],
        "reduced_full_dim_attempts": reduced_qp["full_dim_attempts"],
        "full_qp_attempts": full_qp["attempts"],
        "full_qp_successes": full_qp["successes"],
        "full_qp_failures": full_qp["failures"],
        "full_failed_sqp_iteration": full["failed_sqp_iteration"],
        "full_failed_rows": full["failed_rows"],
    }


def main():
    # Align settings with the validated 1000-candidate configuration.
    base.ROBUST_GAP_TOL = ROBUST_GAP_TOL
    base.LAM = LAM
    base.KAPPA = KAPPA
    base.REDUCED_COST_TOL = REDUCED_COST_TOL
    base.ACTIVE_TOL = ACTIVE_TOL

    problem = load_1000_problem()
    _, _, _, n, _, _, _ = problem

    if n != 1000:
        raise RuntimeError(f"Expected n=1000, but loaded n={n}.")

    print("=" * 94)
    print("FINAL 1000-CANDIDATE BENCHMARK")
    print("Reduced-space active-set HiGHS SQP vs direct full-space HiGHS SQP")
    print("=" * 94)
    print("Mapping: S1000 -> Sigma, A1000 -> Omega")
    print(f"lambda = {LAM}")
    print(f"kappa = {KAPPA}")
    print(f"robust-gap tolerance = {ROBUST_GAP_TOL:.1e}")
    print(f"reduced-cost tolerance = {REDUCED_COST_TOL:.1e}")
    print(f"active tolerance = {ACTIVE_TOL:.1e}")
    print(f"timed paired repeats = {N_REPEATS}")
    print("One warm-up pair is run first and is NOT included in statistics.")
    print()

    # ------------------------------------------------------------------
    # Warm-up: unreported
    # ------------------------------------------------------------------
    print("Warm-up pair...")
    warmup = run_pair(problem, 0)
    print(
        f"  reduced-space: "
        f"{'OK' if warmup['reduced_success'] else 'FAIL'}, "
        f"{warmup['reduced_runtime_s']:.3f}s"
    )
    print(
        f"  full-space:    "
        f"{'OK' if warmup['full_success'] else 'FAIL'}, "
        f"{warmup['full_runtime_s']:.3f}s"
    )
    print("Warm-up discarded from final statistics.")
    print()

    # ------------------------------------------------------------------
    # Timed repeats
    # ------------------------------------------------------------------
    rows = []

    for repeat in range(1, N_REPEATS + 1):
        row = run_pair(problem, repeat)
        rows.append(row)

        print(f"Repeat {repeat:2d}/{N_REPEATS}:")

        if row["reduced_success"]:
            print(
                f"  reduced-space: OK, "
                f"{row['reduced_runtime_s']:.3f}s, "
                f"SQPs={row['reduced_sqp_subproblems']}, "
                f"gap={row['reduced_robust_gap']:.3e}"
            )
        else:
            print("  reduced-space: FAIL")

        if row["full_success"]:
            print(
                f"  full-space:    OK, "
                f"{row['full_runtime_s']:.3f}s, "
                f"SQPs={row['full_sqp_subproblems']}, "
                f"gap={row['full_robust_gap']:.3e}"
            )
        else:
            print(
                f"  full-space:    FAIL at SQP "
                f"{row['full_failed_sqp_iteration']}"
            )

        if row["speedup_full_over_reduced"] is not None:
            print(
                f"  speed-up full/reduced = "
                f"{row['speedup_full_over_reduced']:.3f}x"
            )
            print(
                f"  max|dw|={row['max_abs_w_diff']:.3e}, "
                f"|df|={row['objective_abs_diff']:.3e}"
            )
        else:
            print("  speed-up = N/A")

        print(
            f"  reduced QP dim: median="
            f"{row['reduced_qp_median_dim']}, "
            f"mean={row['reduced_qp_mean_dim']:.3f}, "
            f"full-dim attempts={row['reduced_full_dim_attempts']}, "
            f"QP failures={row['reduced_qp_failures']}"
        )
        print()

    # ------------------------------------------------------------------
    # Detailed results CSV
    # ------------------------------------------------------------------
    results_path = Path("benchmark_1000_final10_results.csv")
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    reduced_successes = [
        row for row in rows if row["reduced_success"]
    ]
    full_successes = [
        row for row in rows if row["full_success"]
    ]
    paired = [
        row for row in rows
        if row["reduced_success"] and row["full_success"]
    ]

    reduced_runtime = stats(
        row["reduced_runtime_s"] for row in reduced_successes
    )
    full_runtime = stats(
        row["full_runtime_s"] for row in full_successes
    )
    paired_speedup = stats(
        row["speedup_full_over_reduced"] for row in paired
    )

    reduced_sqp = stats(
        row["reduced_sqp_subproblems"] for row in reduced_successes
    )
    full_sqp = stats(
        row["full_sqp_subproblems"] for row in full_successes
    )

    reduced_qp_mean_dims = stats(
        row["reduced_qp_mean_dim"] for row in reduced_successes
    )
    reduced_qp_median_dims = stats(
        row["reduced_qp_median_dim"] for row in reduced_successes
    )
    reduced_full_dim_attempts = stats(
        row["reduced_full_dim_attempts"] for row in reduced_successes
    )
    reduced_qp_failures = stats(
        row["reduced_qp_failures"] for row in reduced_successes
    )

    max_solution_difference = (
        max(row["max_abs_w_diff"] for row in paired)
        if paired else None
    )
    max_objective_difference = (
        max(row["objective_abs_diff"] for row in paired)
        if paired else None
    )
    max_z_difference = (
        max(row["z_abs_diff"] for row in paired)
        if paired else None
    )

    mean_dimension_fraction = (
        reduced_qp_mean_dims["mean"] / FULL_QP_DIMENSION
        if reduced_qp_mean_dims["mean"] is not None
        else None
    )
    mean_dimension_reduction_pct = (
        100.0 * (1.0 - mean_dimension_fraction)
        if mean_dimension_fraction is not None
        else None
    )

    summary_row = {
        "problem_candidates": 1000,
        "full_qp_dimension": FULL_QP_DIMENSION,
        "lambda": LAM,
        "kappa": KAPPA,
        "robust_gap_tolerance": ROBUST_GAP_TOL,
        "reduced_cost_tolerance": REDUCED_COST_TOL,
        "active_tolerance": ACTIVE_TOL,
        "timed_repeats": N_REPEATS,
        "reduced_successes": len(reduced_successes),
        "full_successes": len(full_successes),
        "valid_paired_comparisons": len(paired),
        "reduced_runtime_mean_s": reduced_runtime["mean"],
        "reduced_runtime_median_s": reduced_runtime["median"],
        "reduced_runtime_stdev_s": reduced_runtime["stdev"],
        "reduced_runtime_min_s": reduced_runtime["min"],
        "reduced_runtime_max_s": reduced_runtime["max"],
        "full_runtime_mean_s": full_runtime["mean"],
        "full_runtime_median_s": full_runtime["median"],
        "full_runtime_stdev_s": full_runtime["stdev"],
        "full_runtime_min_s": full_runtime["min"],
        "full_runtime_max_s": full_runtime["max"],
        "speedup_mean_full_over_reduced": paired_speedup["mean"],
        "speedup_median_full_over_reduced": paired_speedup["median"],
        "speedup_stdev": paired_speedup["stdev"],
        "speedup_min": paired_speedup["min"],
        "speedup_max": paired_speedup["max"],
        "reduced_mean_sqp_subproblems": reduced_sqp["mean"],
        "full_mean_sqp_subproblems": full_sqp["mean"],
        "reduced_qp_mean_dimension": reduced_qp_mean_dims["mean"],
        "reduced_qp_median_dimension": reduced_qp_median_dims["mean"],
        "mean_dimension_fraction_of_full": mean_dimension_fraction,
        "mean_dimension_reduction_percent": mean_dimension_reduction_pct,
        "mean_reduced_full_dimension_attempts": reduced_full_dim_attempts["mean"],
        "mean_reduced_qp_failures": reduced_qp_failures["mean"],
        "max_solution_difference": max_solution_difference,
        "max_objective_difference": max_objective_difference,
        "max_z_difference": max_z_difference,
    }

    summary_csv_path = Path("benchmark_1000_final10_summary.csv")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_row.keys()),
        )
        writer.writeheader()
        writer.writerow(summary_row)

    # ------------------------------------------------------------------
    # Human-readable final report
    # ------------------------------------------------------------------
    try:
        highs_ver = base.highs_version()
    except Exception:
        highs_ver = "unavailable"

    lines = [
        "=" * 94,
        "FINAL 1000-CANDIDATE BENCHMARK SUMMARY",
        "=" * 94,
        f"Python = {sys.version.split()[0]}",
        f"HiGHS = {highs_ver}",
        f"Platform = {platform.platform()}",
        "Mapping = S1000 -> Sigma, A1000 -> Omega",
        f"Candidates = 1000",
        f"Full QP dimension = {FULL_QP_DIMENSION}",
        f"lambda = {LAM}",
        f"kappa = {KAPPA}",
        f"robust-gap tolerance = {ROBUST_GAP_TOL:.1e}",
        f"reduced-cost tolerance = {REDUCED_COST_TOL:.1e}",
        f"active tolerance = {ACTIVE_TOL:.1e}",
        f"timed repeats = {N_REPEATS}",
        "Warm-up = one unreported pair",
        "",
        "Convergence",
        f"  reduced-space = {len(reduced_successes)}/{N_REPEATS}",
        f"  full-space    = {len(full_successes)}/{N_REPEATS}",
        f"  valid paired comparisons = {len(paired)}",
        "",
        "Reduced-space runtime (s)",
        f"  mean   = {fmt(reduced_runtime['mean'])}",
        f"  median = {fmt(reduced_runtime['median'])}",
        f"  stdev  = {fmt(reduced_runtime['stdev'])}",
        f"  min    = {fmt(reduced_runtime['min'])}",
        f"  max    = {fmt(reduced_runtime['max'])}",
        "",
        "Full-space runtime (s)",
        f"  mean   = {fmt(full_runtime['mean'])}",
        f"  median = {fmt(full_runtime['median'])}",
        f"  stdev  = {fmt(full_runtime['stdev'])}",
        f"  min    = {fmt(full_runtime['min'])}",
        f"  max    = {fmt(full_runtime['max'])}",
        "",
        "Speed-up = full-space runtime / reduced-space runtime",
        f"  mean   = {fmt(paired_speedup['mean'])}x",
        f"  median = {fmt(paired_speedup['median'])}x",
        f"  stdev  = {fmt(paired_speedup['stdev'])}",
        f"  min    = {fmt(paired_speedup['min'])}x",
        f"  max    = {fmt(paired_speedup['max'])}x",
        "",
        "SQP subproblems",
        f"  reduced-space mean = {fmt(reduced_sqp['mean'], 3)}",
        f"  full-space mean    = {fmt(full_sqp['mean'], 3)}",
        "",
        "Reduced-space QP dimensions",
        f"  full QP dimension = {FULL_QP_DIMENSION}",
        f"  mean QP dimension = {fmt(reduced_qp_mean_dims['mean'], 3)}",
        f"  median QP dimension = {fmt(reduced_qp_median_dims['mean'], 3)}",
        f"  mean fraction of full dimension = "
        f"{fmt(mean_dimension_fraction, 6)}",
        f"  mean dimension reduction = "
        f"{fmt(mean_dimension_reduction_pct, 3)}%",
        f"  mean full-dimension fallback attempts = "
        f"{fmt(reduced_full_dim_attempts['mean'], 3)}",
        f"  mean failed QP attempts = "
        f"{fmt(reduced_qp_failures['mean'], 3)}",
        "",
        "Solution agreement across valid paired runs",
        (
            f"  maximum max|w_reduced - w_full| = "
            f"{max_solution_difference:.12e}"
            if max_solution_difference is not None
            else "  maximum solution difference = N/A"
        ),
        (
            f"  maximum objective absolute difference = "
            f"{max_objective_difference:.12e}"
            if max_objective_difference is not None
            else "  maximum objective difference = N/A"
        ),
        (
            f"  maximum |z_reduced - z_full| = "
            f"{max_z_difference:.12e}"
            if max_z_difference is not None
            else "  maximum z difference = N/A"
        ),
        "",
        "Reporting rule",
        "  Runtime speed-up is interpreted only for paired runs in which",
        "  both methods converged. The warm-up pair is excluded from all",
        "  reported timing statistics.",
        "=" * 94,
    ]

    summary_text = "\n".join(lines)
    summary_txt_path = Path("benchmark_1000_final10_summary.txt")
    summary_txt_path.write_text(
        summary_text + "\n",
        encoding="utf-8",
    )

    print()
    print(summary_text)
    print()
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {summary_txt_path}")


if __name__ == "__main__":
    main()
