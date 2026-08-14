#!/usr/bin/env python3
"""
Tolerance-sensitivity benchmark for Robust OCS
==============================================

Compares the SAME two methods used in benchmark_full_vs_active_v2.py:

    1. reduced-space active-set HiGHS SQP
    2. direct full-space HiGHS SQP

for several robust-gap tolerances.

IMPORTANT
---------
Place this file in the RobustOCS repository root together with:

    benchmark_full_vs_active_v2.py

Then run:

    python benchmark_tolerance_sweep.py

The benchmark changes ONLY the outer SQP robust-gap tolerance.  For each
tolerance, both methods use identical problem data and identical numerical
settings except for the reduced-space versus full-space QP strategy.

Outputs
-------
    benchmark_tolerance_results.csv
    benchmark_tolerance_summary.csv
    benchmark_tolerance_summary.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

import benchmark_full_vs_active_v2 as base


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

TOLERANCES = [
    1e-5,
    1e-6,
    5e-7,
    1e-7,
]

N_REPEATS = 10


def safe_stats(values):
    values = [float(v) for v in values if v is not None]
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


def fmt(value, digits=9):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def run_one_pair(problem, tolerance, repeat, tracker):
    """
    Run one reduced-space/full-space pair.  The order alternates between
    repeats to reduce systematic timing bias.
    """
    base.ROBUST_GAP_TOL = tolerance

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

    pair = {
        "tolerance": tolerance,
        "repeat": repeat,
        "reduced_success": reduced["success"],
        "full_success": full["success"],
        "reduced_runtime_s": reduced["runtime_s"],
        "full_runtime_s": full["runtime_s"],
        "reduced_sqp_subproblems": reduced["sqp_subproblems_attempted"],
        "full_sqp_subproblems": full["sqp_subproblems_attempted"],
        "reduced_final_or_last_gap": reduced["robust_gap"],
        "full_final_or_last_gap": full["robust_gap"],
        "reduced_final_or_last_objective": reduced["objective"],
        "full_final_or_last_objective": full["objective"],
        "reduced_nonzero_w": reduced["nonzero_w"],
        "full_nonzero_w": full["nonzero_w"],
        "full_failed_sqp_iteration": full["failed_sqp_iteration"],
        "full_failed_planes": full["failed_planes"],
        "full_failed_rows": full["failed_rows"],
        "full_last_success_gap_signed": full["last_success_gap"],
        "full_failure_type": full["failure_type"],
        "full_failure_model_status": full["failure_model_status"],
        "full_failure_message": full["failure_message"],
        "paired_speedup_full_over_reduced": None,
        "max_abs_w_diff": None,
        "objective_abs_diff": None,
        "z_abs_diff": None,
    }

    # A timing speed-up is valid only when BOTH methods converged.
    if reduced["success"] and full["success"]:
        pair["paired_speedup_full_over_reduced"] = (
            full["runtime_s"] / reduced["runtime_s"]
        )
        pair["max_abs_w_diff"] = float(
            np.max(np.abs(reduced["w"] - full["w"]))
        )
        pair["objective_abs_diff"] = abs(
            reduced["objective"] - full["objective"]
        )
        pair["z_abs_diff"] = abs(reduced["z"] - full["z"])

    return pair


def main():
    # Keep all settings identical to benchmark v2 except the tolerance sweep.
    base.N_REPEATS = N_REPEATS

    problem = base.load_50_problem()

    print("=" * 88)
    print("Robust OCS tolerance-sensitivity benchmark")
    print("Reduced-space HiGHS SQP vs full-space HiGHS SQP")
    print("50 candidates; mapping: S50 -> Sigma, A50 -> Omega")
    print("=" * 88)
    print("Tolerances:", ", ".join(f"{tol:.1e}" for tol in TOLERANCES))
    print(f"Timed repeats per tolerance: {N_REPEATS}")

    detailed_rows = []

    with base.QPAttemptTracker() as tracker:
        # One unreported warm-up pair at the loosest tolerance.
        warm_tol = TOLERANCES[0]
        base.ROBUST_GAP_TOL = warm_tol

        print(f"\nWarm-up at tolerance {warm_tol:.1e}...")
        warm_reduced = base.run_instrumented_sqp(
            base.qp_module.highs_active_set_qp,
            problem,
            "reduced-space",
            0,
            tracker,
        )
        warm_full = base.run_instrumented_sqp(
            base.full_space_highs_qp,
            problem,
            "full-space",
            0,
            tracker,
        )
        print(
            f"  reduced-space: {'OK' if warm_reduced['success'] else 'FAIL'}"
            f", time={warm_reduced['runtime_s']:.6f}s"
        )
        print(
            f"  full-space:    {'OK' if warm_full['success'] else 'FAIL'}"
            f", time={warm_full['runtime_s']:.6f}s"
        )

        tracker.records.clear()

        for tolerance in TOLERANCES:
            print("\n" + "-" * 88)
            print(f"Tolerance = {tolerance:.1e}")
            print("-" * 88)

            for repeat in range(1, N_REPEATS + 1):
                pair = run_one_pair(
                    problem=problem,
                    tolerance=tolerance,
                    repeat=repeat,
                    tracker=tracker,
                )
                detailed_rows.append(pair)

                reduced_text = (
                    f"OK {pair['reduced_runtime_s']:.6f}s, "
                    f"SQPs={pair['reduced_sqp_subproblems']}, "
                    f"gap={pair['reduced_final_or_last_gap']:.3e}"
                    if pair["reduced_success"]
                    else (
                        f"FAIL {pair['reduced_runtime_s']:.6f}s, "
                        f"gap={pair['reduced_final_or_last_gap']}"
                    )
                )

                if pair["full_success"]:
                    full_text = (
                        f"OK {pair['full_runtime_s']:.6f}s, "
                        f"SQPs={pair['full_sqp_subproblems']}, "
                        f"gap={pair['full_final_or_last_gap']:.3e}"
                    )
                else:
                    full_text = (
                        f"FAIL {pair['full_runtime_s']:.6f}s "
                        f"at SQP {pair['full_failed_sqp_iteration']}, "
                        f"last gap={pair['full_last_success_gap_signed']}"
                    )

                print(f"Repeat {repeat:2d}:")
                print(f"  reduced-space: {reduced_text}")
                print(f"  full-space:    {full_text}")

                if pair["paired_speedup_full_over_reduced"] is not None:
                    print(
                        "  speed-up(full/reduced)="
                        f"{pair['paired_speedup_full_over_reduced']:.3f}x, "
                        f"max|dw|={pair['max_abs_w_diff']:.3e}, "
                        f"|df|={pair['objective_abs_diff']:.3e}"
                    )
                else:
                    print("  speed-up: N/A (both methods did not converge)")

    # -----------------------------------------------------------------------
    # Detailed CSV
    # -----------------------------------------------------------------------

    detailed_path = Path("benchmark_tolerance_results.csv")
    with detailed_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(detailed_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    # -----------------------------------------------------------------------
    # Per-tolerance summary
    # -----------------------------------------------------------------------

    summary_rows = []

    for tolerance in TOLERANCES:
        rows = [
            row for row in detailed_rows
            if row["tolerance"] == tolerance
        ]

        reduced_successes = [
            row for row in rows if row["reduced_success"]
        ]
        full_successes = [
            row for row in rows if row["full_success"]
        ]
        full_failures = [
            row for row in rows if not row["full_success"]
        ]

        reduced_runtime_stats = safe_stats(
            row["reduced_runtime_s"]
            for row in reduced_successes
        )
        full_runtime_stats = safe_stats(
            row["full_runtime_s"]
            for row in full_successes
        )
        speedup_stats = safe_stats(
            row["paired_speedup_full_over_reduced"]
            for row in rows
            if row["paired_speedup_full_over_reduced"] is not None
        )
        max_w_diff_stats = safe_stats(
            row["max_abs_w_diff"]
            for row in rows
            if row["max_abs_w_diff"] is not None
        )
        objective_diff_stats = safe_stats(
            row["objective_abs_diff"]
            for row in rows
            if row["objective_abs_diff"] is not None
        )
        reduced_sqp_stats = safe_stats(
            row["reduced_sqp_subproblems"]
            for row in reduced_successes
        )
        full_sqp_stats = safe_stats(
            row["full_sqp_subproblems"]
            for row in full_successes
        )
        failure_iteration_stats = safe_stats(
            row["full_failed_sqp_iteration"]
            for row in full_failures
            if row["full_failed_sqp_iteration"] is not None
        )

        summary_rows.append({
            "tolerance": tolerance,
            "reduced_successes": len(reduced_successes),
            "full_successes": len(full_successes),
            "reduced_mean_runtime_s": reduced_runtime_stats["mean"],
            "reduced_median_runtime_s": reduced_runtime_stats["median"],
            "reduced_stdev_runtime_s": reduced_runtime_stats["stdev"],
            "full_mean_runtime_s": full_runtime_stats["mean"],
            "full_median_runtime_s": full_runtime_stats["median"],
            "full_stdev_runtime_s": full_runtime_stats["stdev"],
            "valid_speedup_pairs": speedup_stats["n"],
            "mean_speedup_full_over_reduced": speedup_stats["mean"],
            "median_speedup_full_over_reduced": speedup_stats["median"],
            "mean_max_abs_w_diff": max_w_diff_stats["mean"],
            "max_max_abs_w_diff": max_w_diff_stats["max"],
            "mean_objective_abs_diff": objective_diff_stats["mean"],
            "max_objective_abs_diff": objective_diff_stats["max"],
            "mean_reduced_sqp_subproblems": reduced_sqp_stats["mean"],
            "mean_full_sqp_subproblems": full_sqp_stats["mean"],
            "mean_full_failure_iteration": failure_iteration_stats["mean"],
        })

    summary_csv_path = Path("benchmark_tolerance_summary.csv")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # -----------------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------------

    lines = [
        "=" * 88,
        "ROBUST OCS TOLERANCE-SENSITIVITY SUMMARY",
        "=" * 88,
        f"Python = {base.sys.version.split()[0]}",
        f"HiGHS  = {base.highs_version()}",
        "Problem = 50 candidates (51 QP variables including z)",
        "Mapping = S50 -> Sigma, A50 -> Omega",
        f"lambda = {base.LAM}",
        f"kappa = {base.KAPPA}",
        f"reduced_cost_tol = {base.REDUCED_COST_TOL}",
        f"active_tol = {base.ACTIVE_TOL}",
        f"repeats per tolerance = {N_REPEATS}",
        "",
    ]

    for row in summary_rows:
        tol = row["tolerance"]
        lines.extend([
            f"Tolerance {tol:.1e}",
            f"  reduced-space convergence = "
            f"{row['reduced_successes']}/{N_REPEATS}",
            f"  full-space convergence    = "
            f"{row['full_successes']}/{N_REPEATS}",
            f"  reduced mean runtime (s)  = "
            f"{fmt(row['reduced_mean_runtime_s'])}",
            f"  full mean runtime (s)     = "
            f"{fmt(row['full_mean_runtime_s'])}",
            f"  valid paired comparisons  = "
            f"{row['valid_speedup_pairs']}",
            f"  mean speed-up full/reduced = "
            f"{fmt(row['mean_speedup_full_over_reduced'], 6)}",
            f"  median speed-up            = "
            f"{fmt(row['median_speedup_full_over_reduced'], 6)}",
            "  max solution difference   = "
            + (
                "N/A"
                if row["max_max_abs_w_diff"] is None
                else f"{row['max_max_abs_w_diff']:.3e}"
            ),
            "  max objective difference  = "
            + (
                "N/A"
                if row["max_objective_abs_diff"] is None
                else f"{row['max_objective_abs_diff']:.3e}"
            ),
            f"  mean reduced SQP solves   = "
            f"{fmt(row['mean_reduced_sqp_subproblems'], 3)}",
            f"  mean full SQP solves      = "
            f"{fmt(row['mean_full_sqp_subproblems'], 3)}",
            f"  mean full failure iter.   = "
            f"{fmt(row['mean_full_failure_iteration'], 3)}",
            "",
        ])

    lines.extend([
        "Interpretation rules",
        "  1. Runtime speed-up is computed only when both methods converge.",
        "  2. A tolerance sweep is a sensitivity analysis, not a replacement for",
        "     the strict 1e-7 experiment.",
        "  3. If the two methods converge to materially different solutions at a",
        "     given tolerance, the timing comparison must be interpreted with",
        "     caution and the solution differences should be reported.",
        "",
        f"Detailed results: {detailed_path}",
        f"Summary CSV:      {summary_csv_path}",
        "=" * 88,
    ])

    summary_text = "\n".join(lines)
    summary_txt_path = Path("benchmark_tolerance_summary.txt")
    summary_txt_path.write_text(
        summary_text + "\n",
        encoding="utf-8",
    )

    print("\n" + summary_text)
    print(f"\nSaved: {detailed_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {summary_txt_path}")


if __name__ == "__main__":
    main()
