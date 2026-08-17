#!/usr/bin/env python3
"""
Robust OCS benchmark v2
=======================

Purpose
-------
Compare:
    1. the current reduced-space active-set HiGHS QP method; and
    2. a full-space HiGHS QP baseline,

while using the SAME instrumented outer SQP loop for both methods.

Unlike v1, this script does NOT crash when a QP solve fails. It records:
    - convergence / failure status;
    - SQP iteration at failure;
    - number of tangent planes and constraint rows at failure;
    - last successfully computed robust gap;
    - runtime to convergence or failure;
    - QP dimensions actually passed to HiGHS;
    - objective / solution differences when both methods converge.

Run from the RobustOCS repository root:
    python benchmark_full_vs_active_v2.py

Outputs:
    benchmark_50_v2_results.csv
    benchmark_50_v2_iterations.csv
    benchmark_50_v2_qp_attempts.csv
    benchmark_50_v2_summary.txt
"""

from __future__ import annotations

import csv
import platform
import sys
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import highspy
import numpy as np
from scipy import sparse

import robustocs as rocs
import robustocs.qp_solver as qp_module


# ---------------------------------------------------------------------------
# Experiment settings: identical to the current 50-candidate test
# ---------------------------------------------------------------------------

N_REPEATS = 10

LAM = 0.5
KAPPA = 1.0

ROBUST_GAP_TOL = 1e-7
REDUCED_COST_TOL = 1e-8
ACTIVE_TOL = 1e-10

MAX_SQP_ITERATIONS = 1000
MAX_QP_ITERATIONS = 1000

FULL_DIMENSION = 51  # 50 candidate contributions + z


# ---------------------------------------------------------------------------
# Matrix / bound helpers
# ---------------------------------------------------------------------------

def bound_array(dimension, value):
    if np.isscalar(value):
        return np.full(dimension, float(value), dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (dimension,):
        raise ValueError(
            f"Bound must be scalar or have shape ({dimension},), got {result.shape}."
        )
    return result.copy()


def symmetric_for_matvec(matrix):
    if sparse.issparse(matrix):
        result = matrix.tocsr().astype(np.float64)
    else:
        result = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))

    difference = result - result.transpose()
    difference.eliminate_zeros()

    if difference.nnz == 0:
        return result

    return (
        result
        + result.transpose()
        - sparse.diags(result.diagonal(), format="csr")
    ).tocsr()


# ---------------------------------------------------------------------------
# Full-space QP baseline
# ---------------------------------------------------------------------------

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
    """
    Solve the supplied convex QP directly with ALL variables in HiGHS.

    This intentionally bypasses:
        - full-variable LP initialisation;
        - reduced-variable selection;
        - reduced-cost expansion.

    The QP model itself is still built using the same helper currently used by
    the repository's active-set solver.
    """
    if sparse.issparse(hessian):
        hessian_csr = hessian.tocsr().astype(np.float64)
    else:
        hessian_csr = sparse.csr_matrix(
            np.asarray(hessian, dtype=np.float64)
        )

    linear_cost_array = np.asarray(linear_cost, dtype=np.float64)
    constraint_matrix_array = np.asarray(
        constraint_matrix, dtype=np.float64
    )

    if rhs is not None:
        if row_lower is not None or row_upper is not None:
            raise ValueError(
                "Pass either rhs or row_lower/row_upper, not both."
            )
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


# ---------------------------------------------------------------------------
# QP-attempt tracker
# ---------------------------------------------------------------------------

class QPAttemptTracker:
    """
    Temporarily wraps qp_module._solve_reduced_highs_qp so that every LP/QP
    attempt can be recorded without modifying the repository source files.
    """

    def __init__(self):
        self.records = []
        self.current_method = None
        self.current_repeat = None
        self.current_sqp_iteration = None
        self._original = None

    def __enter__(self):
        self._original = qp_module._solve_reduced_highs_qp

        def tracked(*args, **kwargs):
            active_indices = np.asarray(
                kwargs.get("active_indices", args[7] if len(args) > 7 else []),
                dtype=int,
            )
            use_quadratic = kwargs.get(
                "use_quadratic", args[8] if len(args) > 8 else None
            )
            qp_iteration = kwargs.get(
                "iteration", args[12] if len(args) > 12 else None
            )
            row_lower = kwargs.get(
                "row_lower", args[3] if len(args) > 3 else np.array([])
            )

            record = {
                "method": self.current_method,
                "repeat": self.current_repeat,
                "sqp_iteration": self.current_sqp_iteration,
                "inner_qp_iteration": qp_iteration,
                "problem_type": "QP" if use_quadratic else "LP",
                "dimension": int(active_indices.size),
                "rows": int(np.asarray(row_lower).size),
                "success": False,
                "error": "",
            }

            try:
                result = self._original(*args, **kwargs)
                record["success"] = True
                return result
            except Exception as exc:
                record["error"] = (
                    f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"
                )
                raise
            finally:
                self.records.append(record)

        qp_module._solve_reduced_highs_qp = tracked
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        qp_module._solve_reduced_highs_qp = self._original


# ---------------------------------------------------------------------------
# Instrumented outer SQP
# ---------------------------------------------------------------------------

def run_instrumented_sqp(
    qp_solver,
    problem,
    method_name,
    repeat,
    tracker,
):
    """
    Mirror the current highs_robust_genetics_sqp outer loop, while recording
    convergence and failure diagnostics. Both benchmark methods use this exact
    same outer loop.
    """
    sigma, mubar, omega, dimension, sires, dams, names = problem

    if sparse.issparse(sigma):
        sigma_csr = sigma.tocsr().astype(np.float64)
    else:
        sigma_csr = sparse.csr_matrix(
            np.asarray(sigma, dtype=np.float64)
        )

    if sparse.issparse(omega):
        omega_csr = omega.tocsr().astype(np.float64)
    else:
        omega_csr = sparse.csr_matrix(
            np.asarray(omega, dtype=np.float64)
        )

    mubar = np.asarray(mubar, dtype=np.float64)
    sires = list(sires)
    dams = list(dams)

    sigma_full = symmetric_for_matvec(sigma_csr)
    omega_full = symmetric_for_matvec(omega_csr)

    qp_dimension = dimension + 1

    hessian = sparse.block_diag(
        (
            LAM * sigma_csr,
            sparse.csr_matrix((1, 1), dtype=np.float64),
        ),
        format="csr",
    )

    linear_cost = np.concatenate(
        (
            -mubar,
            np.array([KAPPA], dtype=np.float64),
        )
    )

    lower_w = bound_array(dimension, 0.0)
    upper_w = bound_array(dimension, 1.0)

    lower_x = np.concatenate(
        (lower_w, np.array([0.0], dtype=np.float64))
    )
    upper_x = np.concatenate(
        (
            upper_w,
            np.array([highspy.kHighsInf], dtype=np.float64),
        )
    )

    base_matrix = np.zeros((2, qp_dimension), dtype=np.float64)
    base_matrix[0, sires] = 1.0
    base_matrix[1, dams] = 1.0

    base_row_lower = np.full(2, 0.5, dtype=np.float64)
    base_row_upper = np.full(2, 0.5, dtype=np.float64)

    plane_rows = []
    iteration_records = []

    last_success_iteration = None
    last_success_gap = None
    last_success_objective = None
    last_success_z = None
    last_success_alpha = None
    last_success_w = None

    start = perf_counter()

    for sqp_iteration in range(MAX_SQP_ITERATIONS):
        if plane_rows:
            constraint_matrix = np.vstack([base_matrix, *plane_rows])
            number_of_planes = len(plane_rows)
            row_lower = np.concatenate(
                (
                    base_row_lower,
                    np.zeros(number_of_planes, dtype=np.float64),
                )
            )
            row_upper = np.concatenate(
                (
                    base_row_upper,
                    np.full(
                        number_of_planes,
                        highspy.kHighsInf,
                        dtype=np.float64,
                    ),
                )
            )
        else:
            constraint_matrix = base_matrix.copy()
            row_lower = base_row_lower.copy()
            row_upper = base_row_upper.copy()

        tracker.current_method = method_name
        tracker.current_repeat = repeat
        tracker.current_sqp_iteration = sqp_iteration

        try:
            solution, minimum_objective = qp_solver(
                hessian=hessian,
                linear_cost=linear_cost,
                constraint_matrix=constraint_matrix,
                row_lower=row_lower,
                row_upper=row_upper,
                lower_bound=lower_x,
                upper_bound=upper_x,
                time_limit=None,
                model_output="",
                debug=False,
                max_iterations=MAX_QP_ITERATIONS,
                reduced_cost_tol=REDUCED_COST_TOL,
                active_tol=ACTIVE_TOL,
                full_qp_fallback=True,
            )
        except Exception as exc:
            elapsed = perf_counter() - start

            model_status = getattr(exc, "model_status", None)
            if model_status is None and exc.__cause__ is not None:
                model_status = getattr(exc.__cause__, "model_status", None)

            return {
                "method": method_name,
                "repeat": repeat,
                "success": False,
                "runtime_s": elapsed,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc).replace("\n", " "),
                "failure_model_status": str(model_status) if model_status else "",
                "failed_sqp_iteration": sqp_iteration,
                "failed_planes": len(plane_rows),
                "failed_rows": int(row_lower.size),
                "last_success_iteration": last_success_iteration,
                "last_success_gap": last_success_gap,
                "last_success_objective": last_success_objective,
                "last_success_z": last_success_z,
                "last_success_alpha": last_success_alpha,
                "sqp_subproblems_attempted": sqp_iteration + 1,
                "w": last_success_w,
                "z": last_success_z,
                "objective": last_success_objective,
                "robust_gap": (
                    abs(last_success_gap)
                    if last_success_gap is not None
                    else None
                ),
                "nonzero_w": (
                    int(np.sum(last_success_w > ACTIVE_TOL))
                    if last_success_w is not None
                    else None
                ),
                "iteration_records": iteration_records,
            }

        w_star = np.asarray(solution[:dimension], dtype=np.float64)
        z_star = float(solution[-1])

        uncertainty_squared = float(
            w_star.transpose() @ (omega_full @ w_star)
        )

        if uncertainty_squared < -ROBUST_GAP_TOL:
            elapsed = perf_counter() - start
            return {
                "method": method_name,
                "repeat": repeat,
                "success": False,
                "runtime_s": elapsed,
                "failure_type": "RuntimeError",
                "failure_message": (
                    "Uncertainty quadratic form negative beyond tolerance: "
                    f"{uncertainty_squared:.12e}"
                ),
                "failure_model_status": "",
                "failed_sqp_iteration": sqp_iteration,
                "failed_planes": len(plane_rows),
                "failed_rows": int(row_lower.size),
                "last_success_iteration": last_success_iteration,
                "last_success_gap": last_success_gap,
                "last_success_objective": last_success_objective,
                "last_success_z": last_success_z,
                "last_success_alpha": last_success_alpha,
                "sqp_subproblems_attempted": sqp_iteration + 1,
                "w": last_success_w,
                "z": last_success_z,
                "objective": last_success_objective,
                "robust_gap": (
                    abs(last_success_gap)
                    if last_success_gap is not None
                    else None
                ),
                "nonzero_w": (
                    int(np.sum(last_success_w > ACTIVE_TOL))
                    if last_success_w is not None
                    else None
                ),
                "iteration_records": iteration_records,
            }

        alpha = sqrt(max(0.0, uncertainty_squared))
        robust_gap = alpha - z_star

        coancestry = float(
            w_star.transpose() @ (sigma_full @ w_star)
        )
        objective_value = float(
            mubar.transpose() @ w_star
            - 0.5 * LAM * coancestry
            - KAPPA * z_star
        )

        iteration_records.append({
            "method": method_name,
            "repeat": repeat,
            "sqp_iteration": sqp_iteration,
            "planes": len(plane_rows),
            "rows": int(row_lower.size),
            "z": z_star,
            "alpha": alpha,
            "gap": robust_gap,
            "abs_gap": abs(robust_gap),
            "robust_objective": objective_value,
            "qp_objective_maximisation_sign": -float(minimum_objective),
        })

        last_success_iteration = sqp_iteration
        last_success_gap = robust_gap
        last_success_objective = objective_value
        last_success_z = z_star
        last_success_alpha = alpha
        last_success_w = w_star.copy()

        if abs(robust_gap) <= ROBUST_GAP_TOL:
            elapsed = perf_counter() - start
            return {
                "method": method_name,
                "repeat": repeat,
                "success": True,
                "runtime_s": elapsed,
                "failure_type": "",
                "failure_message": "",
                "failure_model_status": "",
                "failed_sqp_iteration": None,
                "failed_planes": None,
                "failed_rows": None,
                "last_success_iteration": sqp_iteration,
                "last_success_gap": robust_gap,
                "last_success_objective": objective_value,
                "last_success_z": z_star,
                "last_success_alpha": alpha,
                "sqp_subproblems_attempted": sqp_iteration + 1,
                "w": w_star,
                "z": z_star,
                "objective": objective_value,
                "robust_gap": abs(robust_gap),
                "nonzero_w": int(np.sum(w_star > ACTIVE_TOL)),
                "iteration_records": iteration_records,
            }

        if alpha <= ACTIVE_TOL:
            elapsed = perf_counter() - start
            return {
                "method": method_name,
                "repeat": repeat,
                "success": False,
                "runtime_s": elapsed,
                "failure_type": "RuntimeError",
                "failure_message": (
                    "SQP cannot form tangent plane because alpha is "
                    "numerically zero."
                ),
                "failure_model_status": "",
                "failed_sqp_iteration": sqp_iteration,
                "failed_planes": len(plane_rows),
                "failed_rows": int(row_lower.size),
                "last_success_iteration": sqp_iteration,
                "last_success_gap": robust_gap,
                "last_success_objective": objective_value,
                "last_success_z": z_star,
                "last_success_alpha": alpha,
                "sqp_subproblems_attempted": sqp_iteration + 1,
                "w": w_star,
                "z": z_star,
                "objective": objective_value,
                "robust_gap": abs(robust_gap),
                "nonzero_w": int(np.sum(w_star > ACTIVE_TOL)),
                "iteration_records": iteration_records,
            }

        uncertainty_gradient = np.asarray(
            omega_full @ w_star,
            dtype=np.float64,
        ).reshape(-1) / alpha

        plane_row = np.concatenate(
            (
                -uncertainty_gradient,
                np.array([1.0], dtype=np.float64),
            )
        )

        if not np.all(np.isfinite(plane_row)):
            elapsed = perf_counter() - start
            return {
                "method": method_name,
                "repeat": repeat,
                "success": False,
                "runtime_s": elapsed,
                "failure_type": "RuntimeError",
                "failure_message": "SQP generated a non-finite tangent plane.",
                "failure_model_status": "",
                "failed_sqp_iteration": sqp_iteration,
                "failed_planes": len(plane_rows),
                "failed_rows": int(row_lower.size),
                "last_success_iteration": sqp_iteration,
                "last_success_gap": robust_gap,
                "last_success_objective": objective_value,
                "last_success_z": z_star,
                "last_success_alpha": alpha,
                "sqp_subproblems_attempted": sqp_iteration + 1,
                "w": w_star,
                "z": z_star,
                "objective": objective_value,
                "robust_gap": abs(robust_gap),
                "nonzero_w": int(np.sum(w_star > ACTIVE_TOL)),
                "iteration_records": iteration_records,
            }

        plane_rows.append(plane_row)

    elapsed = perf_counter() - start
    return {
        "method": method_name,
        "repeat": repeat,
        "success": False,
        "runtime_s": elapsed,
        "failure_type": "MaxIterations",
        "failure_message": (
            f"SQP did not converge after {MAX_SQP_ITERATIONS} iterations."
        ),
        "failure_model_status": "",
        "failed_sqp_iteration": MAX_SQP_ITERATIONS,
        "failed_planes": len(plane_rows),
        "failed_rows": 2 + len(plane_rows),
        "last_success_iteration": last_success_iteration,
        "last_success_gap": last_success_gap,
        "last_success_objective": last_success_objective,
        "last_success_z": last_success_z,
        "last_success_alpha": last_success_alpha,
        "sqp_subproblems_attempted": MAX_SQP_ITERATIONS,
        "w": last_success_w,
        "z": last_success_z,
        "objective": last_success_objective,
        "robust_gap": (
            abs(last_success_gap)
            if last_success_gap is not None
            else None
        ),
        "nonzero_w": (
            int(np.sum(last_success_w > ACTIVE_TOL))
            if last_success_w is not None
            else None
        ),
        "iteration_records": iteration_records,
    }


# ---------------------------------------------------------------------------
# Loading / summary helpers
# ---------------------------------------------------------------------------

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
            "Run this script from the RobustOCS repository root.\n"
            "Missing files:\n" + "\n".join(missing)
        )

    return rocs.load_problem(
        sigma_filename=str(data_dir / "A50.txt"),
        mu_filename=str(data_dir / "EBV50.txt"),
        omega_filename=str(data_dir / "S50.txt"),
        sex_filename=str(data_dir / "SEX50.txt"),
        issparse=True,
    )


def safe_stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None

    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def format_stats(label, stats):
    if stats is None:
        return [
            label,
            "  no applicable observations",
        ]
    return [
        label,
        f"  n      = {stats['n']}",
        f"  mean   = {stats['mean']:.9f}",
        f"  median = {stats['median']:.9f}",
        f"  stdev  = {stats['stdev']:.9f}",
        f"  min    = {stats['min']:.9f}",
        f"  max    = {stats['max']:.9f}",
    ]


def highs_version():
    try:
        return highspy.Highs().version()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    problem = load_50_problem()

    print("=" * 80)
    print("Robust OCS benchmark v2: 50 candidates")
    print("Reduced-space HiGHS SQP vs full-space HiGHS SQP")
    print("Mapping: A50 -> Sigma, S50 -> Omega")
    print("=" * 80)

    all_results = []
    all_iteration_rows = []

    with QPAttemptTracker() as tracker:
        # Warm-up runs: recorded in console only, not included in timed summary.
        print("\nWarm-up: reduced-space...")
        warm_active = run_instrumented_sqp(
            qp_module.highs_active_set_qp,
            problem,
            "reduced-space",
            0,
            tracker,
        )
        print(
            "  "
            + (
                f"converged in {warm_active['runtime_s']:.6f}s"
                if warm_active["success"]
                else (
                    f"failed at SQP {warm_active['failed_sqp_iteration']} "
                    f"after {warm_active['runtime_s']:.6f}s"
                )
            )
        )

        print("Warm-up: full-space...")
        warm_full = run_instrumented_sqp(
            full_space_highs_qp,
            problem,
            "full-space",
            0,
            tracker,
        )
        print(
            "  "
            + (
                f"converged in {warm_full['runtime_s']:.6f}s"
                if warm_full["success"]
                else (
                    f"failed at SQP {warm_full['failed_sqp_iteration']} "
                    f"(planes={warm_full['failed_planes']}, "
                    f"rows={warm_full['failed_rows']}) "
                    f"after {warm_full['runtime_s']:.6f}s; "
                    f"last successful gap="
                    f"{warm_full['last_success_gap']}"
                )
            )
        )

        # Exclude warm-up QP attempts from exported timing analysis.
        tracker.records.clear()

        print("\nTimed runs:")
        for repeat in range(1, N_REPEATS + 1):
            if repeat % 2 == 1:
                active = run_instrumented_sqp(
                    qp_module.highs_active_set_qp,
                    problem,
                    "reduced-space",
                    repeat,
                    tracker,
                )
                full = run_instrumented_sqp(
                    full_space_highs_qp,
                    problem,
                    "full-space",
                    repeat,
                    tracker,
                )
            else:
                full = run_instrumented_sqp(
                    full_space_highs_qp,
                    problem,
                    "full-space",
                    repeat,
                    tracker,
                )
                active = run_instrumented_sqp(
                    qp_module.highs_active_set_qp,
                    problem,
                    "reduced-space",
                    repeat,
                    tracker,
                )

            if active["success"] and full["success"]:
                max_w_diff = float(
                    np.max(np.abs(active["w"] - full["w"]))
                )
                objective_diff = abs(
                    active["objective"] - full["objective"]
                )
                z_diff = abs(active["z"] - full["z"])
                paired_speedup = (
                    full["runtime_s"] / active["runtime_s"]
                )
            else:
                max_w_diff = None
                objective_diff = None
                z_diff = None
                paired_speedup = None

            for result in (active, full):
                row = {
                    "repeat": repeat,
                    "method": result["method"],
                    "success": result["success"],
                    "runtime_s": result["runtime_s"],
                    "sqp_subproblems_attempted": result[
                        "sqp_subproblems_attempted"
                    ],
                    "final_or_last_abs_gap": result["robust_gap"],
                    "final_or_last_objective": result["objective"],
                    "nonzero_w": result["nonzero_w"],
                    "failed_sqp_iteration": result[
                        "failed_sqp_iteration"
                    ],
                    "failed_planes": result["failed_planes"],
                    "failed_rows": result["failed_rows"],
                    "last_success_iteration": result[
                        "last_success_iteration"
                    ],
                    "last_success_gap_signed": result[
                        "last_success_gap"
                    ],
                    "failure_type": result["failure_type"],
                    "failure_model_status": result[
                        "failure_model_status"
                    ],
                    "failure_message": result["failure_message"],
                    "paired_speedup_full_over_reduced": (
                        paired_speedup
                        if result["method"] == "reduced-space"
                        else ""
                    ),
                    "paired_max_abs_w_diff": (
                        max_w_diff
                        if result["method"] == "reduced-space"
                        else ""
                    ),
                    "paired_objective_abs_diff": (
                        objective_diff
                        if result["method"] == "reduced-space"
                        else ""
                    ),
                    "paired_z_abs_diff": (
                        z_diff
                        if result["method"] == "reduced-space"
                        else ""
                    ),
                }
                all_results.append(row)

                all_iteration_rows.extend(
                    result["iteration_records"]
                )

            active_text = (
                f"OK {active['runtime_s']:.6f}s, "
                f"SQPs={active['sqp_subproblems_attempted']}, "
                f"gap={active['robust_gap']:.3e}"
                if active["success"]
                else (
                    f"FAIL {active['runtime_s']:.6f}s at SQP "
                    f"{active['failed_sqp_iteration']}, "
                    f"rows={active['failed_rows']}, "
                    f"last gap={active['last_success_gap']}"
                )
            )

            full_text = (
                f"OK {full['runtime_s']:.6f}s, "
                f"SQPs={full['sqp_subproblems_attempted']}, "
                f"gap={full['robust_gap']:.3e}"
                if full["success"]
                else (
                    f"FAIL {full['runtime_s']:.6f}s at SQP "
                    f"{full['failed_sqp_iteration']}, "
                    f"rows={full['failed_rows']}, "
                    f"last gap={full['last_success_gap']}"
                )
            )

            print(f"Repeat {repeat:2d}:")
            print(f"  reduced-space: {active_text}")
            print(f"  full-space:    {full_text}")

            if paired_speedup is not None:
                print(
                    f"  paired speed-up = {paired_speedup:.3f}x, "
                    f"max|dw|={max_w_diff:.3e}, "
                    f"|df|={objective_diff:.3e}"
                )
            else:
                print(
                    "  paired speed-up = N/A "
                    "(both methods must converge for a valid speed comparison)"
                )

        qp_attempt_rows = tracker.records.copy()

    # -----------------------------------------------------------------------
    # Export detailed CSV files
    # -----------------------------------------------------------------------

    results_path = Path("benchmark_50_v2_results.csv")
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_results)

    iterations_path = Path("benchmark_50_v2_iterations.csv")
    if all_iteration_rows:
        with iterations_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(all_iteration_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(all_iteration_rows)

    qp_attempts_path = Path("benchmark_50_v2_qp_attempts.csv")
    if qp_attempt_rows:
        with qp_attempts_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(qp_attempt_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(qp_attempt_rows)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------

    active_rows = [
        row for row in all_results
        if row["method"] == "reduced-space"
    ]
    full_rows = [
        row for row in all_results
        if row["method"] == "full-space"
    ]

    active_successes = [row for row in active_rows if row["success"]]
    full_successes = [row for row in full_rows if row["success"]]

    active_failures = [row for row in active_rows if not row["success"]]
    full_failures = [row for row in full_rows if not row["success"]]

    active_success_runtime = safe_stats(
        row["runtime_s"] for row in active_successes
    )
    full_success_runtime = safe_stats(
        row["runtime_s"] for row in full_successes
    )
    full_failure_runtime = safe_stats(
        row["runtime_s"] for row in full_failures
    )

    valid_speedups = [
        row["paired_speedup_full_over_reduced"]
        for row in active_rows
        if row["paired_speedup_full_over_reduced"] not in ("", None)
    ]
    speedup_stats = safe_stats(valid_speedups)

    failed_full_iterations = safe_stats(
        row["failed_sqp_iteration"]
        for row in full_failures
        if row["failed_sqp_iteration"] is not None
    )

    failed_full_rows = safe_stats(
        row["failed_rows"]
        for row in full_failures
        if row["failed_rows"] is not None
    )

    last_full_failure_gaps = safe_stats(
        abs(row["last_success_gap_signed"])
        for row in full_failures
        if row["last_success_gap_signed"] is not None
    )

    # QP dimensions: exclude LPs; show dimensions actually passed to HiGHS.
    active_qp_attempts = [
        row for row in qp_attempt_rows
        if row["method"] == "reduced-space"
        and row["problem_type"] == "QP"
    ]
    full_qp_attempts = [
        row for row in qp_attempt_rows
        if row["method"] == "full-space"
        and row["problem_type"] == "QP"
    ]

    active_qp_dimension_stats = safe_stats(
        row["dimension"] for row in active_qp_attempts
    )
    full_qp_dimension_stats = safe_stats(
        row["dimension"] for row in full_qp_attempts
    )

    active_full_dimension_attempts = sum(
        1 for row in active_qp_attempts
        if row["dimension"] == FULL_DIMENSION
    )

    lines = [
        "=" * 80,
        "BENCHMARK 50 V2 SUMMARY",
        "=" * 80,
        "",
        "Experiment environment",
        f"  Python = {sys.version.split()[0]}",
        f"  HiGHS  = {highs_version()}",
        f"  OS     = {platform.platform()}",
        f"  Machine= {platform.machine()}",
        f"  Processor = {platform.processor()}",
        "",
        "Problem settings",
        "  candidates = 50",
        "  complete QP dimension = 51 (50 w variables + z)",
        "  mapping = A50 -> Sigma, S50 -> Omega",
        f"  lambda = {LAM}",
        f"  kappa = {KAPPA}",
        f"  robust_gap_tol = {ROBUST_GAP_TOL}",
        f"  reduced_cost_tol = {REDUCED_COST_TOL}",
        f"  active_tol = {ACTIVE_TOL}",
        f"  timed repeats = {N_REPEATS}",
        "",
        "Convergence counts",
        f"  reduced-space successes = {len(active_successes)}/{N_REPEATS}",
        f"  reduced-space failures  = {len(active_failures)}/{N_REPEATS}",
        f"  full-space successes    = {len(full_successes)}/{N_REPEATS}",
        f"  full-space failures     = {len(full_failures)}/{N_REPEATS}",
        "",
    ]

    lines.extend(format_stats(
        "Reduced-space runtime for converged runs (s):",
        active_success_runtime,
    ))
    lines.append("")
    lines.extend(format_stats(
        "Full-space runtime for converged runs (s):",
        full_success_runtime,
    ))
    lines.append("")
    lines.extend(format_stats(
        "Full-space time to failure (s):",
        full_failure_runtime,
    ))
    lines.append("")
    lines.extend(format_stats(
        "Valid paired speed-up = full / reduced:",
        speedup_stats,
    ))
    lines.append("")
    lines.extend(format_stats(
        "Full-space failed SQP iteration:",
        failed_full_iterations,
    ))
    lines.append("")
    lines.extend(format_stats(
        "Full-space rows at failure:",
        failed_full_rows,
    ))
    lines.append("")
    lines.extend(format_stats(
        "Absolute robust gap at last successful full-space iteration:",
        last_full_failure_gaps,
    ))
    lines.append("")
    lines.extend(format_stats(
        "Reduced-space QP dimensions actually passed to HiGHS:",
        active_qp_dimension_stats,
    ))
    lines.append(
        f"  full-dimension QP attempts inside reduced-space method = "
        f"{active_full_dimension_attempts}"
    )
    lines.append("")
    lines.extend(format_stats(
        "Full-space QP dimensions actually passed to HiGHS:",
        full_qp_dimension_stats,
    ))
    lines.extend([
        "",
        "Important interpretation rule",
        "  A speed-up is reported only when BOTH methods converge in the same repeat.",
        "  If full-space fails, its runtime is time-to-failure, NOT a valid competing",
        "  solution time, so no speed-up should be inferred from that pair.",
        "",
        "Files",
        f"  detailed run results = {results_path}",
        f"  per-SQP-iteration results = {iterations_path}",
        f"  individual LP/QP attempts = {qp_attempts_path}",
        "=" * 80,
    ])

    summary_text = "\n".join(lines)
    summary_path = Path("benchmark_50_v2_summary.txt")
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    print("\n" + summary_text)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {results_path}")
    print(f"Saved: {iterations_path}")
    print(f"Saved: {qp_attempts_path}")


if __name__ == "__main__":
    main()
