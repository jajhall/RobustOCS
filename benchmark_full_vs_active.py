#!/usr/bin/env python3
from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import highspy
import numpy as np
from scipy import sparse

import robustocs as rocs
import robustocs.qp_solver as qp_module
import robustocs.solvers_restructured as solvers

N_REPEATS = 10
LAM = 0.5
KAPPA = 1.0
ROBUST_GAP_TOL = 1e-7
REDUCED_COST_TOL = 1e-8
ACTIVE_TOL = 1e-10
MAX_SQP_ITERATIONS = 1000
MAX_QP_ITERATIONS = 1000


def full_symmetric(matrix):
    if sparse.issparse(matrix):
        result = matrix.tocsr().astype(np.float64)
    else:
        result = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    difference = result - result.transpose()
    difference.eliminate_zeros()
    if difference.nnz == 0:
        return result
    return (
        result + result.transpose()
        - sparse.diags(result.diagonal(), format="csr")
    ).tocsr()


def full_space_highs_qp(
    hessian,
    linear_cost,
    constraint_matrix,
    rhs=None,
    lower_bound=0.0,
    upper_bound=highspy.kHighsInf,
    row_lower=None,
    row_upper=None,
    time_limit=None,
    model_output="",
    debug=False,
    max_iterations=1000,
    reduced_cost_tol=1e-8,
    active_tol=1e-10,
    full_qp_fallback=True,
):
    if sparse.issparse(hessian):
        hessian_csr = hessian.tocsr().astype(np.float64)
    else:
        hessian_csr = sparse.csr_matrix(np.asarray(hessian, dtype=np.float64))

    linear_cost_array = np.asarray(linear_cost, dtype=np.float64)
    constraint_matrix_array = np.asarray(constraint_matrix, dtype=np.float64)

    if rhs is not None:
        if row_lower is not None or row_upper is not None:
            raise ValueError("Pass either rhs or row_lower/row_upper, not both.")
        rhs_array = np.asarray(rhs, dtype=np.float64)
        row_lower_array = rhs_array.copy()
        row_upper_array = rhs_array.copy()
    else:
        if row_lower is None or row_upper is None:
            raise ValueError(
                "row_lower and row_upper are required when rhs is omitted."
            )
        row_lower_array = np.asarray(row_lower, dtype=np.float64)
        row_upper_array = np.asarray(row_upper, dtype=np.float64)

    dimension = linear_cost_array.size
    lower = qp_module._bound_array(dimension, lower_bound)
    upper = qp_module._bound_array(dimension, upper_bound)

    qp_module._validate_qp_data(
        hessian=hessian_csr,
        linear_cost=linear_cost_array,
        constraint_matrix=constraint_matrix_array,
        row_lower=row_lower_array,
        row_upper=row_upper_array,
        lower=lower,
        upper=upper,
    )

    all_indices = np.arange(dimension, dtype=int)
    solution, objective, _, _ = qp_module._solve_reduced_highs_qp(
        hessian=hessian_csr,
        linear_cost=linear_cost_array,
        constraint_matrix=constraint_matrix_array,
        row_lower=row_lower_array,
        row_upper=row_upper_array,
        lower=lower,
        upper=upper,
        active_indices=all_indices,
        use_quadratic=True,
        time_limit=time_limit,
        model_output=model_output,
        debug=debug,
        iteration=0,
    )
    return solution, objective


def load_50_problem():
    data_dir = Path("examples") / "50"
    required_files = [
        data_dir / "S50.txt",
        data_dir / "A50.txt",
        data_dir / "EBV50.txt",
        data_dir / "SEX50.txt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run this script from the RobustOCS repository root. Missing:\n"
            + "\n".join(missing)
        )

    return rocs.load_problem(
        sigma_filename=str(data_dir / "S50.txt"),
        mu_filename=str(data_dir / "EBV50.txt"),
        omega_filename=str(data_dir / "A50.txt"),
        sex_filename=str(data_dir / "SEX50.txt"),
        issparse=True,
    )


def solve_once(qp_solver, problem):
    sigma, mubar, omega, dimension, sires, dams, names = problem

    original_qp_solver = solvers.highs_active_set_qp
    qp_calls = 0

    def counted_qp_solver(*args, **kwargs):
        nonlocal qp_calls
        qp_calls += 1
        return qp_solver(*args, **kwargs)

    solvers.highs_active_set_qp = counted_qp_solver

    try:
        start = perf_counter()
        w, z, objective = solvers.highs_robust_genetics_sqp(
            sigma=sigma,
            mubar=mubar,
            omega=omega,
            sires=sires,
            dams=dams,
            lam=LAM,
            kappa=KAPPA,
            dimension=dimension,
            lower_bound=0.0,
            upper_bound=1.0,
            max_iterations=MAX_SQP_ITERATIONS,
            robust_gap_tol=ROBUST_GAP_TOL,
            qp_max_iterations=MAX_QP_ITERATIONS,
            reduced_cost_tol=REDUCED_COST_TOL,
            active_tol=ACTIVE_TOL,
            debug=False,
        )
        elapsed = perf_counter() - start
    finally:
        solvers.highs_active_set_qp = original_qp_solver

    omega_full = full_symmetric(omega)
    exact_uncertainty = sqrt(max(0.0, float(w @ (omega_full @ w))))

    return {
        "time": elapsed,
        "w": np.asarray(w, dtype=np.float64),
        "z": float(z),
        "objective": float(objective),
        "sqp_subproblems": qp_calls,
        "robust_gap": abs(float(z) - exact_uncertainty),
        "nonzero_w": int(np.sum(np.asarray(w) > ACTIVE_TOL)),
    }


def summary_stats(values):
    values = list(values)
    return {
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    problem = load_50_problem()

    print("=" * 80)
    print("Robust OCS benchmark: 50 candidates")
    print("Reduced-space HiGHS SQP vs full-space HiGHS SQP")
    print("Mapping: S50 -> Sigma, A50 -> Omega")
    print("=" * 80)

    active_solver = qp_module.highs_active_set_qp
    full_solver = full_space_highs_qp

    print("\nWarm-up: reduced-space...")
    solve_once(active_solver, problem)
    print("Warm-up: full-space...")
    solve_once(full_solver, problem)
    print("Warm-up complete.\n")

    rows = []
    active_results = []
    full_results = []

    for repeat in range(1, N_REPEATS + 1):
        if repeat % 2 == 1:
            active = solve_once(active_solver, problem)
            full = solve_once(full_solver, problem)
        else:
            full = solve_once(full_solver, problem)
            active = solve_once(active_solver, problem)

        active_results.append(active)
        full_results.append(full)

        max_w_diff = float(np.max(np.abs(active["w"] - full["w"])))
        objective_diff = abs(active["objective"] - full["objective"])
        z_diff = abs(active["z"] - full["z"])
        paired_speedup = full["time"] / active["time"]

        row = {
            "repeat": repeat,
            "active_time_s": active["time"],
            "full_time_s": full["time"],
            "paired_speedup_full_over_active": paired_speedup,
            "active_objective": active["objective"],
            "full_objective": full["objective"],
            "objective_abs_diff": objective_diff,
            "max_abs_w_diff": max_w_diff,
            "z_abs_diff": z_diff,
            "active_robust_gap": active["robust_gap"],
            "full_robust_gap": full["robust_gap"],
            "active_sqp_subproblems": active["sqp_subproblems"],
            "full_sqp_subproblems": full["sqp_subproblems"],
            "active_nonzero_w": active["nonzero_w"],
            "full_nonzero_w": full["nonzero_w"],
        }
        rows.append(row)

        print(
            f"Repeat {repeat:2d}: "
            f"active={active['time']:.6f}s, "
            f"full={full['time']:.6f}s, "
            f"speed-up={paired_speedup:.3f}x, "
            f"max|dw|={max_w_diff:.3e}, "
            f"|df|={objective_diff:.3e}"
        )

    csv_path = Path("benchmark_50_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    active_time_stats = summary_stats(r["time"] for r in active_results)
    full_time_stats = summary_stats(r["time"] for r in full_results)
    speedup_stats = summary_stats(
        row["paired_speedup_full_over_active"] for row in rows
    )

    max_w_diff_all = max(row["max_abs_w_diff"] for row in rows)
    max_obj_diff_all = max(row["objective_abs_diff"] for row in rows)
    max_z_diff_all = max(row["z_abs_diff"] for row in rows)
    mean_speedup_ratio = full_time_stats["mean"] / active_time_stats["mean"]

    summary_lines = [
        "=" * 80,
        "BENCHMARK SUMMARY",
        "=" * 80,
        f"Repeats (timed): {N_REPEATS}",
        "",
        "Reduced-space active-set SQP runtime (s):",
        f"  mean   = {active_time_stats['mean']:.9f}",
        f"  median = {active_time_stats['median']:.9f}",
        f"  stdev  = {active_time_stats['stdev']:.9f}",
        f"  min    = {active_time_stats['min']:.9f}",
        f"  max    = {active_time_stats['max']:.9f}",
        "",
        "Full-space SQP runtime (s):",
        f"  mean   = {full_time_stats['mean']:.9f}",
        f"  median = {full_time_stats['median']:.9f}",
        f"  stdev  = {full_time_stats['stdev']:.9f}",
        f"  min    = {full_time_stats['min']:.9f}",
        f"  max    = {full_time_stats['max']:.9f}",
        "",
        "Speed-up = full-space time / reduced-space time:",
        f"  ratio of mean times = {mean_speedup_ratio:.6f}x",
        f"  mean paired speed-up = {speedup_stats['mean']:.6f}x",
        f"  median paired speed-up = {speedup_stats['median']:.6f}x",
        "",
        "Agreement between methods:",
        f"  maximum max|w_active - w_full| = {max_w_diff_all:.12e}",
        f"  maximum |objective_active - objective_full| = {max_obj_diff_all:.12e}",
        f"  maximum |z_active - z_full| = {max_z_diff_all:.12e}",
        "",
        "SQP subproblems in final timed run:",
        f"  reduced-space = {active_results[-1]['sqp_subproblems']}",
        f"  full-space    = {full_results[-1]['sqp_subproblems']}",
        "",
        "Non-zero candidate contributions in final timed run:",
        f"  reduced-space = {active_results[-1]['nonzero_w']}",
        f"  full-space    = {full_results[-1]['nonzero_w']}",
        "",
        "Interpretation:",
        "  speed-up > 1 : reduced-space method is faster.",
        "  speed-up < 1 : full-space method is faster.",
        "=" * 80,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = Path("benchmark_50_summary.txt")
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    print(f"\nSaved detailed results to: {csv_path}")
    print(f"Saved summary to:          {summary_path}")


if __name__ == "__main__":
    main()
