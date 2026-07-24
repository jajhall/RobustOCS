# -*- coding: utf-8 -*-
"""General active-set quadratic-programming solver based on HiGHS.

The solver in this module is deliberately independent of Optimal Contribution
Selection.  It solves convex QP subproblems of the form

    minimise    c^T x + 1/2 x^T Q x
    subject to  M x = m,
                lower <= x <= upper.

For the robust OCS SQP method, ``x`` is the combined vector ``[w, z]`` and
each SQP iteration can append another row to ``M`` and another entry to ``m``.
"""

from time import time

import highspy
import numpy as np
import numpy.typing as npt
from scipy import sparse

__all__ = ["highs_active_set_qp"]


Bound = float | list[float] | npt.NDArray[np.float64]

# Keep the numerical QP solves more accurate than the active-set reduced-cost
# threshold, so variables are not added or rejected because of solver noise.
_HIGHS_KKT_TOL = 1e-9


def _bound_array(dimension: int, value: Bound) -> npt.NDArray[np.float64]:
    """Convert a scalar/list/array bound into a float64 vector."""

    if np.isscalar(value):
        return np.full(dimension, float(value), dtype=np.float64)

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (dimension,):
        raise ValueError(
            f"Bound must be scalar or have shape ({dimension},), "
            f"got {result.shape}."
        )
    return result.copy()


def _remaining_time(
    time_limit: float | None,
    start_time: float | None
) -> float | None:
    """Return the unused part of an overall time limit."""

    if time_limit is None:
        return None
    if start_time is None:
        return time_limit

    remaining = time_limit - (time() - start_time)
    if remaining <= 0:
        raise RuntimeError(
            f"HiGHS hit the time limit of {time_limit} seconds without "
            "reaching optimality."
        )
    return remaining


def _symmetric_matrix_for_matvec(
    matrix: sparse.spmatrix
) -> sparse.csr_matrix:
    """Return a full symmetric CSR matrix for gradient calculations.

    HiGHS can receive a Hessian stored as a single triangle.  A complete
    symmetric matrix is nevertheless needed when evaluating ``Q @ x`` in the
    reduced-cost formula.
    """

    matrix = matrix.tocsr()
    difference = matrix - matrix.transpose()
    difference.eliminate_zeros()

    if difference.nnz == 0:
        return matrix

    return (
        matrix
        + matrix.transpose()
        - sparse.diags(matrix.diagonal(), format="csr")
    ).tocsr()


def _validate_qp_data(
    hessian: sparse.spmatrix,
    linear_cost: npt.NDArray[np.float64],
    constraint_matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64]
) -> None:
    """Validate dimensions and simple bound conditions."""

    dimension = linear_cost.size
    number_of_constraints = rhs.size

    if hessian.shape != (dimension, dimension):
        raise ValueError(
            f"hessian must have shape ({dimension}, {dimension}), "
            f"got {hessian.shape}."
        )

    if constraint_matrix.shape != (number_of_constraints, dimension):
        raise ValueError(
            "constraint_matrix must have shape "
            f"({number_of_constraints}, {dimension}), "
            f"got {constraint_matrix.shape}."
        )

    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError("Bounds must have one entry per decision variable.")

    if np.any(lower > upper):
        raise ValueError("Every lower bound must be no greater than its upper bound.")


def _solve_reduced_highs_qp(
    hessian: sparse.csr_matrix,
    linear_cost: npt.NDArray[np.float64],
    constraint_matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    active_indices: npt.NDArray[np.int_],
    use_quadratic: bool,
    time_limit: float | None,
    model_output: str,
    debug: bool,
    iteration: int
) -> tuple[
    npt.NDArray[np.float64],
    float,
    npt.NDArray[np.float64],
    tuple[int, int]
]:
    """Solve the LP/QP restricted to ``active_indices``."""

    active_indices = np.asarray(active_indices, dtype=int)
    if active_indices.ndim != 1 or active_indices.size == 0:
        raise ValueError("active_indices must be a non-empty one-dimensional array.")

    dimension = linear_cost.size
    reduced_dimension = active_indices.size
    number_of_constraints = rhs.size

    hessian_reduced = hessian[active_indices, :][:, active_indices].tocsr()
    cost_reduced = linear_cost[active_indices]
    lower_reduced = lower[active_indices]
    upper_reduced = upper[active_indices]

    matrix_reduced = sparse.csr_matrix(
        constraint_matrix[:, active_indices],
        dtype=np.float64
    )

    model = highspy.HighsModel()
    model.lp_.model_name_ = "general-reduced-active-set-qp"
    model.lp_.num_col_ = reduced_dimension
    model.lp_.num_row_ = number_of_constraints
    model.lp_.col_cost_ = cost_reduced.tolist()
    model.lp_.col_lower_ = lower_reduced.tolist()
    model.lp_.col_upper_ = upper_reduced.tolist()

    if use_quadratic:
        model.hessian_.format_ = highspy.HessianFormat.kSquare
        model.hessian_.dim_ = reduced_dimension
        model.hessian_.start_ = hessian_reduced.indptr.tolist()
        model.hessian_.index_ = hessian_reduced.indices.tolist()
        model.hessian_.value_ = hessian_reduced.data.tolist()

    model.lp_.row_lower_ = rhs.tolist()
    model.lp_.row_upper_ = rhs.tolist()
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = matrix_reduced.indptr.tolist()
    model.lp_.a_matrix_.index_ = matrix_reduced.indices.tolist()
    model.lp_.a_matrix_.value_ = matrix_reduced.data.tolist()

    highs = highspy.Highs()
    highs.setOptionValue("kkt_tolerance", _HIGHS_KKT_TOL)
    if not debug:
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("log_to_console", False)

    pass_status = highs.passModel(model)

    if model_output:
        problem_type = "qp" if use_quadratic else "lp"
        highs.writeModel(
            f"{model_output}_{problem_type}_iter_{iteration}.mps"
        )

    if pass_status == highspy.HighsStatus.kError:
        raise ValueError(
            f"highs.passModel failed with status {highs.getModelStatus()}."
        )

    if time_limit is not None:
        highs.setOptionValue("time_limit", time_limit)

    run_status = highs.run()
    model_status = highs.getModelStatus()

    if debug:
        print(
            f"\nGeneral QP iteration {iteration}: "
            f"full dimension = {dimension}, "
            f"reduced dimension = {reduced_dimension}, "
            f"constraints = {number_of_constraints}, "
            f"reduced Hessian = {hessian_reduced.shape}, "
            f"active indices = {active_indices.tolist()}"
        )
        highs.writeSolution("", 1)

    if run_status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS failed with status {model_status}.")
    if model_status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(
            "HiGHS did not achieve optimality; "
            f"model status is {model_status}."
        )

    highs_solution = highs.getSolution()
    reduced_solution = np.asarray(
        highs_solution.col_value,
        dtype=np.float64
    )

    full_solution = np.zeros(dimension, dtype=np.float64)
    full_solution[active_indices] = reduced_solution

    objective_value = float(highs.getInfo().objective_function_value)
    row_dual = np.asarray(highs_solution.row_dual, dtype=np.float64)

    return (
        full_solution,
        objective_value,
        row_dual,
        hessian_reduced.shape
    )


def highs_active_set_qp(
    hessian: npt.NDArray[np.float64] | sparse.spmatrix,
    linear_cost: npt.NDArray[np.float64],
    constraint_matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    lower_bound: Bound = 0.0,
    upper_bound: Bound = highspy.kHighsInf,
    time_limit: float | None = None,
    model_output: str = "",
    debug: bool = False,
    max_iterations: int = 1000,
    reduced_cost_tol: float = 1e-8,
    active_tol: float = 1e-10
) -> tuple[npt.NDArray[np.float64], float]:
    """Solve a convex equality-constrained QP using a reduced active set.

    The mathematical problem is

    ``min  linear_cost.T @ x + 0.5 * x.T @ hessian @ x``

    subject to

    ``constraint_matrix @ x = rhs`` and
    ``lower_bound <= x <= upper_bound``.

    The routine contains no sire/dam-specific logic.  For robust OCS SQP,
    supply ``x = [w, z]``.  Adding a new SQP constraint means appending a row
    to ``constraint_matrix`` and the corresponding value to ``rhs`` before
    calling this method again.

    Returns
    -------
    ndarray
        Complete solution vector ``x``.  For robust OCS this is ``[w, z]``.
    float
        Minimum objective value in the minimisation convention above.
    """

    if sparse.issparse(hessian):
        hessian_csr = hessian.tocsr().astype(np.float64)
    else:
        hessian_csr = sparse.csr_matrix(
            np.asarray(hessian, dtype=np.float64)
        )

    linear_cost_array = np.asarray(linear_cost, dtype=np.float64)
    constraint_matrix_array = np.asarray(
        constraint_matrix,
        dtype=np.float64
    )
    rhs_array = np.asarray(rhs, dtype=np.float64)

    if linear_cost_array.ndim != 1:
        raise ValueError("linear_cost must be one-dimensional.")
    if rhs_array.ndim != 1:
        raise ValueError("rhs must be one-dimensional.")
    if constraint_matrix_array.ndim != 2:
        raise ValueError("constraint_matrix must be two-dimensional.")

    dimension = linear_cost_array.size
    lower = _bound_array(dimension, lower_bound)
    upper = _bound_array(dimension, upper_bound)

    _validate_qp_data(
        hessian=hessian_csr,
        linear_cost=linear_cost_array,
        constraint_matrix=constraint_matrix_array,
        rhs=rhs_array,
        lower=lower,
        upper=upper
    )

    # The current reduced-column strategy removes inactive variables by fixing
    # them at zero.  Therefore zero must be a valid lower-bound value for every
    # variable that may be inactive, including the SQP z variable.
    if np.any(np.abs(lower) > active_tol):
        raise ValueError(
            "The reduced active-set QP currently requires every lower bound "
            "to be zero."
        )

    start_time = time() if time_limit is not None else None
    all_indices = np.arange(dimension, dtype=int)

    # Step 0: solve the linear relaxation over all x = [w, z] variables.
    lp_solution, lp_objective, _, _ = _solve_reduced_highs_qp(
        hessian=hessian_csr,
        linear_cost=linear_cost_array,
        constraint_matrix=constraint_matrix_array,
        rhs=rhs_array,
        lower=lower,
        upper=upper,
        active_indices=all_indices,
        use_quadratic=False,
        time_limit=_remaining_time(time_limit, start_time),
        model_output=model_output,
        debug=debug,
        iteration=-1
    )

    active_set = lp_solution > active_tol

    # If the linear relaxation is optimised at x = 0, a positive-semidefinite
    # quadratic term cannot improve the objective away from zero.
    if not np.any(active_set):
        return lp_solution, lp_objective

    hessian_for_gradient = _symmetric_matrix_for_matvec(hessian_csr)

    for iteration in range(max_iterations):
        active_indices = np.flatnonzero(active_set)

        solution, objective_value, row_dual, reduced_hessian_shape = (
            _solve_reduced_highs_qp(
                hessian=hessian_csr,
                linear_cost=linear_cost_array,
                constraint_matrix=constraint_matrix_array,
                rhs=rhs_array,
                lower=lower,
                upper=upper,
                active_indices=active_indices,
                use_quadratic=True,
                time_limit=_remaining_time(time_limit, start_time),
                model_output=model_output,
                debug=debug,
                iteration=iteration
            )
        )

        # HiGHS minimisation reduced cost at a zero lower bound:
        #     r = c + Qx - M^T y.
        reduced_cost = (
            linear_cost_array
            + hessian_for_gradient @ solution
            - constraint_matrix_array.transpose() @ row_dual
        )

        inactive_set = ~active_set
        entering_set = (
            inactive_set
            & (upper > active_tol)
            & (reduced_cost < -reduced_cost_tol)
        )

        if debug:
            if np.any(inactive_set):
                min_inactive = float(np.min(reduced_cost[inactive_set]))
                max_inactive = float(np.max(reduced_cost[inactive_set]))
            else:
                min_inactive = None
                max_inactive = None

            print(
                f"general active-set iteration {iteration}: "
                f"active variables = {np.sum(active_set)}, "
                f"reduced Hessian = {reduced_hessian_shape}, "
                f"min inactive reduced cost = {min_inactive}, "
                f"max inactive reduced cost = {max_inactive}"
            )
            if np.any(entering_set):
                print(
                    "entering variables = "
                    f"{np.flatnonzero(entering_set).tolist()}"
                )

        if not np.any(entering_set):
            return solution, objective_value

        active_set[entering_set] = True

    raise RuntimeError(
        "General active-set QP solver did not converge after "
        f"{max_iterations} iterations."
    )
