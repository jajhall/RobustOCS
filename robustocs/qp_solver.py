# -*- coding: utf-8 -*-
"""General active-set quadratic-programming solver based on HiGHS.

The solver in this module is deliberately independent of Optimal Contribution
Selection.  It solves convex QP subproblems of the form

    minimise    c^T x + 1/2 x^T Q x
    subject to  row_lower <= M x <= row_upper,
                lower <= x <= upper.

Equality constraints are represented by identical row lower and upper bounds.
For robust OCS, ``x`` is the combined vector ``[w, z]``.  The two sex-balance
rows are equalities, while every SQP iteration appends a tangent-plane
inequality for the uncertainty term.
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


class _HighsSubproblemError(RuntimeError):
    """Internal exception carrying the HiGHS model status."""

    def __init__(
        self,
        message: str,
        model_status: highspy.HighsModelStatus
    ) -> None:
        super().__init__(message)
        self.model_status = model_status


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
    row_lower: npt.NDArray[np.float64],
    row_upper: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64]
) -> None:
    """Validate QP dimensions and variable/row bounds."""

    dimension = linear_cost.size
    number_of_constraints = row_lower.size

    if hessian.shape != (dimension, dimension):
        raise ValueError(
            f"hessian must have shape ({dimension}, {dimension}), "
            f"got {hessian.shape}."
        )

    if row_upper.shape != (number_of_constraints,):
        raise ValueError(
            "row_lower and row_upper must have the same one-dimensional "
            "shape."
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
        raise ValueError(
            "Every variable lower bound must be no greater than its upper "
            "bound."
        )

    if np.any(row_lower > row_upper):
        raise ValueError(
            "Every row lower bound must be no greater than its upper bound."
        )


def _enlarge_degenerate_initial_set(
    active_set: npt.NDArray[np.bool_],
    lp_solution: npt.NDArray[np.float64],
    hessian: sparse.csr_matrix,
    linear_cost: npt.NDArray[np.float64],
    constraint_matrix: npt.NDArray[np.float64],
    row_lower: npt.NDArray[np.float64],
    row_upper: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    active_tol: float
) -> npt.NDArray[np.bool_]:
    """Add a variable when the LP active set has no free QP direction.

    The LP used to initialise the variable set can return a vertex with exactly
    as many positive variables as independent tight rows.  In that case the
    reduced feasible region is a single point.  Some versions/configurations of
    the HiGHS active-set QP solver return ``kSolveError`` for such a
    zero-dimensional reduced QP.

    Adding one admissible inactive variable is mathematically harmless: the
    variable is merely made available to the reduced QP and may still receive
    value zero.  It also creates at least one possible reduced direction.

    The candidate is selected using the QP stationarity expression evaluated
    at the LP point without row multipliers,

        c + Q x,

    with the smallest value preferred for this minimisation problem.
    """

    active_set = np.asarray(active_set, dtype=bool).copy()
    activity = constraint_matrix @ lp_solution

    equality_rows = np.isclose(
        row_lower,
        row_upper,
        atol=active_tol,
        rtol=0.0,
    )
    tight_lower = np.isfinite(row_lower) & np.isclose(
        activity,
        row_lower,
        atol=active_tol,
        rtol=0.0,
    )
    tight_upper = np.isfinite(row_upper) & np.isclose(
        activity,
        row_upper,
        atol=active_tol,
        rtol=0.0,
    )
    tight_rows = equality_rows | tight_lower | tight_upper

    while True:
        active_indices = np.flatnonzero(active_set)

        if active_indices.size == 0:
            rank = 0
        elif np.any(tight_rows):
            active_matrix = constraint_matrix[
                tight_rows, :
            ][:, active_indices]
            rank = int(np.linalg.matrix_rank(active_matrix))
        else:
            rank = 0

        # A strictly larger number of active variables than independent tight
        # rows gives the reduced QP at least one potential feasible direction.
        if active_indices.size > rank:
            return active_set

        candidates = np.flatnonzero(
            (~active_set) & (upper > active_tol)
        )
        if candidates.size == 0:
            # All admissible variables are already present.  The caller will
            # handle the resulting fixed feasible point without asking HiGHS
            # to solve a zero-dimensional QP.
            return active_set

        q_gradient_without_duals = (
            linear_cost + hessian @ lp_solution
        )
        entering_index = int(
            candidates[
                np.argmin(q_gradient_without_duals[candidates])
            ]
        )
        active_set[entering_index] = True


def _solve_reduced_highs_qp(
    hessian: sparse.csr_matrix,
    linear_cost: npt.NDArray[np.float64],
    constraint_matrix: npt.NDArray[np.float64],
    row_lower: npt.NDArray[np.float64],
    row_upper: npt.NDArray[np.float64],
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
    number_of_constraints = row_lower.size

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

    model.lp_.row_lower_ = row_lower.tolist()
    model.lp_.row_upper_ = row_upper.tolist()
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

    if (
        run_status == highspy.HighsStatus.kError
        or model_status != highspy.HighsModelStatus.kOptimal
    ):
        status_text = highs.modelStatusToString(model_status)
        raise _HighsSubproblemError(
            "HiGHS did not solve the reduced "
            f"{'QP' if use_quadratic else 'LP'} to optimality: "
            f"{status_text} ({model_status}). "
            f"Reduced dimension={reduced_dimension}, "
            f"rows={number_of_constraints}, "
            f"active indices={active_indices.tolist()}.",
            model_status=model_status,
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

    if (
        not np.all(np.isfinite(reduced_solution))
        or not np.isfinite(objective_value)
        or not np.all(np.isfinite(row_dual))
    ):
        raise _HighsSubproblemError(
            "HiGHS returned non-finite values for the reduced "
            f"{'QP' if use_quadratic else 'LP'}. "
            f"Reduced dimension={reduced_dimension}, "
            f"rows={number_of_constraints}.",
            model_status=model_status,
        )

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
    rhs: npt.NDArray[np.float64] | None = None,
    lower_bound: Bound = 0.0,
    upper_bound: Bound = highspy.kHighsInf,
    row_lower: npt.NDArray[np.float64] | None = None,
    row_upper: npt.NDArray[np.float64] | None = None,
    time_limit: float | None = None,
    model_output: str = "",
    debug: bool = False,
    max_iterations: int = 1000,
    reduced_cost_tol: float = 1e-8,
    active_tol: float = 1e-10,
    full_qp_fallback: bool = True
) -> tuple[npt.NDArray[np.float64], float]:
    """Solve a convex linearly constrained QP using a reduced active set.

    The mathematical problem is

    ``min  linear_cost.T @ x + 0.5 * x.T @ hessian @ x``

    subject to

    ``row_lower <= constraint_matrix @ x <= row_upper`` and
    ``lower_bound <= x <= upper_bound``.

    For equality-only problems, pass ``rhs`` and omit ``row_lower`` and
    ``row_upper``.  This is equivalent to setting both row bounds equal to
    ``rhs`` and preserves the original interface used by standard OCS.

    For robust OCS SQP, use ``x = [w, z]``.  The sire/dam contribution rows
    have identical row bounds of 0.5, while each tangent plane has lower bound
    zero and upper bound positive infinity.

    If a highly degenerate reduced QP is reported as unbounded or otherwise
    fails numerically, ``full_qp_fallback=True`` retries the identical bounded
    convex QP with all variables before reporting failure.

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

    if linear_cost_array.ndim != 1:
        raise ValueError("linear_cost must be one-dimensional.")
    if constraint_matrix_array.ndim != 2:
        raise ValueError("constraint_matrix must be two-dimensional.")

    # Backwards-compatible equality interface: Mx = rhs.
    if rhs is not None:
        if row_lower is not None or row_upper is not None:
            raise ValueError(
                "Pass either rhs for equalities or row_lower/row_upper for "
                "general rows, not both."
            )
        rhs_array = np.asarray(rhs, dtype=np.float64)
        if rhs_array.ndim != 1:
            raise ValueError("rhs must be one-dimensional.")
        row_lower_array = rhs_array.copy()
        row_upper_array = rhs_array.copy()
    else:
        if row_lower is None or row_upper is None:
            raise ValueError(
                "When rhs is omitted, both row_lower and row_upper are "
                "required."
            )
        row_lower_array = np.asarray(row_lower, dtype=np.float64)
        row_upper_array = np.asarray(row_upper, dtype=np.float64)
        if row_lower_array.ndim != 1 or row_upper_array.ndim != 1:
            raise ValueError(
                "row_lower and row_upper must be one-dimensional."
            )

    dimension = linear_cost_array.size
    lower = _bound_array(dimension, lower_bound)
    upper = _bound_array(dimension, upper_bound)

    _validate_qp_data(
        hessian=hessian_csr,
        linear_cost=linear_cost_array,
        constraint_matrix=constraint_matrix_array,
        row_lower=row_lower_array,
        row_upper=row_upper_array,
        lower=lower,
        upper=upper
    )

    # Inactive variables are removed from the reduced model and therefore
    # implicitly fixed at zero.  Zero must consequently be an admissible lower
    # bound for every variable that may be inactive, including z.
    if np.any(np.abs(lower) > active_tol):
        raise ValueError(
            "The reduced active-set QP currently requires every variable "
            "lower bound to be zero."
        )

    start_time = time() if time_limit is not None else None
    all_indices = np.arange(dimension, dtype=int)

    # Step 0: solve the LP relaxation over all variables.  The positive
    # components form the initial reduced variable set.
    lp_solution, lp_objective, _, _ = _solve_reduced_highs_qp(
        hessian=hessian_csr,
        linear_cost=linear_cost_array,
        constraint_matrix=constraint_matrix_array,
        row_lower=row_lower_array,
        row_upper=row_upper_array,
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

    if not np.any(active_set):
        return lp_solution, lp_objective

    # Avoid asking the HiGHS QP active-set solver to solve a reduced problem
    # whose tight constraints fix every active variable.  The paper's
    # three-candidate example produces exactly this corner case after the LP:
    # one positive sire and one positive dam with two independent equalities.
    active_set = _enlarge_degenerate_initial_set(
        active_set=active_set,
        lp_solution=lp_solution,
        hessian=hessian_csr,
        linear_cost=linear_cost_array,
        constraint_matrix=constraint_matrix_array,
        row_lower=row_lower_array,
        row_upper=row_upper_array,
        upper=upper,
        active_tol=active_tol,
    )

    hessian_for_gradient = _symmetric_matrix_for_matvec(hessian_csr)

    # Always solve the QP after LP initialisation.  Even when the LP point is
    # a vertex and all variables are present, inequalities that are tight at
    # the LP solution are allowed to become inactive in the QP.  Therefore the
    # LP point must never be returned merely because the currently tight rows
    # have full rank.

    for iteration in range(max_iterations):
        active_indices = np.flatnonzero(active_set)

        try:
            (
                solution,
                objective_value,
                row_dual,
                reduced_hessian_shape,
            ) = _solve_reduced_highs_qp(
                hessian=hessian_csr,
                linear_cost=linear_cost_array,
                constraint_matrix=constraint_matrix_array,
                row_lower=row_lower_array,
                row_upper=row_upper_array,
                lower=lower,
                upper=upper,
                active_indices=active_indices,
                use_quadratic=True,
                time_limit=_remaining_time(time_limit, start_time),
                model_output=model_output,
                debug=debug,
                iteration=iteration,
            )
        except _HighsSubproblemError as reduced_error:
            # A heavily constrained reduced model can be numerically
            # degenerate even though the original bounded convex QP is well
            # posed.  In that case solve the same QP once with every variable.
            # This remains part of this QP solver; it is only a robust fallback
            # from the reduced formulation to the full formulation.
            if not full_qp_fallback or active_indices.size == dimension:
                raise

            if debug:
                print(
                    "Reduced QP was not solved reliably "
                    f"({reduced_error.model_status}). "
                    "Retrying the same subproblem with all variables."
                )

            try:
                (
                    full_solution,
                    full_objective,
                    _,
                    _,
                ) = _solve_reduced_highs_qp(
                    hessian=hessian_csr,
                    linear_cost=linear_cost_array,
                    constraint_matrix=constraint_matrix_array,
                    row_lower=row_lower_array,
                    row_upper=row_upper_array,
                    lower=lower,
                    upper=upper,
                    active_indices=all_indices,
                    use_quadratic=True,
                    time_limit=_remaining_time(time_limit, start_time),
                    model_output=model_output,
                    debug=debug,
                    iteration=iteration,
                )
            except _HighsSubproblemError as full_error:
                raise RuntimeError(
                    "Both the reduced QP and the full-dimensional fallback "
                    "failed. Reduced status: "
                    f"{reduced_error.model_status}; full status: "
                    f"{full_error.model_status}."
                ) from full_error

            return full_solution, full_objective

        # HiGHS minimisation reduced cost at a zero lower bound:
        #     r = c + Qx - M^T y.
        # The same stationarity expression applies to equality and inequality
        # rows; HiGHS supplies the corresponding signed row duals.
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

