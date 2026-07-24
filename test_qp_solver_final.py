"""Verification tests for robustocs.qp_solver.highs_active_set_qp.

Run from the repository root with:

    python test_qp_solver.py

The tests compare the reduced active-set implementation against:
1. Problems with analytically known solutions.
2. A direct full-dimensional HiGHS QP formulation.
3. Random strictly convex equality-constrained QPs.
"""

from __future__ import annotations

import highspy
import numpy as np
import numpy.typing as npt
from scipy import sparse

from robustocs.qp_solver import highs_active_set_qp


TOL_X = 1e-7
TOL_OBJ = 1e-8
TOL_FEAS = 1e-8
TOL_KKT = 1e-7
SOLVER_KKT_TOL = 1e-9


def objective(
    x: npt.NDArray[np.float64],
    q: sparse.spmatrix,
    c: npt.NDArray[np.float64],
) -> float:
    """Evaluate c^T x + 0.5 x^T Q x independently."""

    return float(c @ x + 0.5 * x @ (q @ x))


def solve_full_highs_qp(
    q: npt.NDArray[np.float64] | sparse.spmatrix,
    c: npt.NDArray[np.float64],
    m_matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Solve the same QP directly with every variable present in HiGHS."""

    q_csr = sparse.csr_matrix(q, dtype=np.float64)
    m_csr = sparse.csr_matrix(m_matrix, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    model = highspy.HighsModel()
    model.lp_.model_name_ = "full-reference-qp"
    model.lp_.num_col_ = c.size
    model.lp_.num_row_ = rhs.size
    model.lp_.col_cost_ = c.tolist()
    model.lp_.col_lower_ = lower.tolist()
    model.lp_.col_upper_ = upper.tolist()

    model.hessian_.format_ = highspy.HessianFormat.kSquare
    model.hessian_.dim_ = c.size
    model.hessian_.start_ = q_csr.indptr.tolist()
    model.hessian_.index_ = q_csr.indices.tolist()
    model.hessian_.value_ = q_csr.data.tolist()

    model.lp_.row_lower_ = rhs.tolist()
    model.lp_.row_upper_ = rhs.tolist()
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = m_csr.indptr.tolist()
    model.lp_.a_matrix_.index_ = m_csr.indices.tolist()
    model.lp_.a_matrix_.value_ = m_csr.data.tolist()

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("log_to_console", False)
    # Ask HiGHS for tighter KKT accuracy than the test threshold.
    option_status = highs.setOptionValue(
        "kkt_tolerance",
        SOLVER_KKT_TOL,
    )
    if option_status == highspy.HighsStatus.kError:
        raise RuntimeError(
            "HiGHS could not set kkt_tolerance. "
            "Check the installed highspy version."
        )

    pass_status = highs.passModel(model)
    if pass_status == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS could not receive the full reference model.")

    run_status = highs.run()
    status = highs.getModelStatus()
    if run_status == highspy.HighsStatus.kError:
        raise RuntimeError(f"Full HiGHS solve failed with status {status}.")
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"Full reference QP was not optimal: {status}.")

    solution = highs.getSolution()
    x = np.asarray(solution.col_value, dtype=np.float64)
    row_dual = np.asarray(solution.row_dual, dtype=np.float64)
    col_dual = np.asarray(solution.col_dual, dtype=np.float64)
    obj = float(highs.getInfo().objective_function_value)
    return x, obj, row_dual, col_dual


def assert_feasible(
    x: npt.NDArray[np.float64],
    m_matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> None:
    """Check equality constraints and variable bounds."""

    equality_residual = float(np.max(np.abs(m_matrix @ x - rhs)))
    lower_violation = float(np.max(np.maximum(lower - x, 0.0)))
    upper_violation = float(np.max(np.maximum(x - upper, 0.0)))

    assert equality_residual <= TOL_FEAS, (
        f"Equality residual too large: {equality_residual:.3e}"
    )
    assert lower_violation <= TOL_FEAS, (
        f"Lower-bound violation too large: {lower_violation:.3e}"
    )
    assert upper_violation <= TOL_FEAS, (
        f"Upper-bound violation too large: {upper_violation:.3e}"
    )


def assert_full_kkt(
    x: npt.NDArray[np.float64],
    q: sparse.spmatrix,
    c: npt.NDArray[np.float64],
    m_matrix: npt.NDArray[np.float64],
    row_dual: npt.NDArray[np.float64],
    col_dual: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> None:
    """Check HiGHS' KKT/reduced-cost sign convention."""

    reconstructed = c + q @ x - m_matrix.T @ row_dual
    dual_residual = reconstructed - col_dual
    worst_index = int(np.argmax(np.abs(dual_residual)))
    dual_difference = float(np.abs(dual_residual[worst_index]))

    # The mathematical tolerance remains TOL_KKT.  Add only a tiny
    # scale-aware IEEE-754 roundoff allowance, so a value such as
    # 1.000000011e-7 is not rejected against a nominal tolerance of 1e-7.
    dual_scale = max(
        1.0,
        float(np.max(np.abs(reconstructed))),
        float(np.max(np.abs(col_dual))),
    )
    roundoff_slack = (
        100.0
        * np.finfo(np.float64).eps
        * dual_scale
    )
    allowed_kkt_error = TOL_KKT + roundoff_slack

    assert dual_difference <= allowed_kkt_error, (
        "Reconstructed reduced costs do not match HiGHS column duals: "
        f"max residual={dual_difference:.16e} at variable {worst_index}; "
        f"allowed={allowed_kkt_error:.16e}; "
        f"reconstructed={reconstructed[worst_index]:.16e}, "
        f"col_dual={col_dual[worst_index]:.16e}"
    )

    at_lower = x <= lower + allowed_kkt_error
    at_upper = x >= upper - allowed_kkt_error
    interior = ~(at_lower | at_upper)

    if np.any(at_lower):
        minimum_lower_rc = float(np.min(reconstructed[at_lower]))
        assert minimum_lower_rc >= -allowed_kkt_error, (
            "A variable at its lower bound has a negative reduced cost: "
            f"{minimum_lower_rc:.16e}"
        )

    if np.any(at_upper):
        maximum_upper_rc = float(np.max(reconstructed[at_upper]))
        assert maximum_upper_rc <= allowed_kkt_error, (
            "A variable at its upper bound has a positive reduced cost: "
            f"{maximum_upper_rc:.16e}"
        )

    if np.any(interior):
        maximum_interior_rc = float(
            np.max(np.abs(reconstructed[interior]))
        )
        assert maximum_interior_rc <= allowed_kkt_error, (
            "An interior variable has a nonzero reduced cost: "
            f"{maximum_interior_rc:.16e}"
        )


def compare_active_with_full(
    q: sparse.spmatrix,
    c: npt.NDArray[np.float64],
    m_matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    """Run both formulations and perform all common checks."""

    active_x, active_obj = highs_active_set_qp(
        hessian=q,
        linear_cost=c,
        constraint_matrix=m_matrix,
        rhs=rhs,
        lower_bound=lower,
        upper_bound=upper,
        reduced_cost_tol=1e-8,
        active_tol=1e-11,
        max_iterations=1000,
        debug=False,
    )

    full_x, full_obj, row_dual, col_dual = solve_full_highs_qp(
        q=q,
        c=c,
        m_matrix=m_matrix,
        rhs=rhs,
        lower=lower,
        upper=upper,
    )

    assert_feasible(active_x, m_matrix, rhs, lower, upper)
    assert_feasible(full_x, m_matrix, rhs, lower, upper)

    active_obj_recalculated = objective(active_x, q, c)
    full_obj_recalculated = objective(full_x, q, c)

    assert abs(active_obj - active_obj_recalculated) <= TOL_OBJ
    assert abs(full_obj - full_obj_recalculated) <= TOL_OBJ
    assert abs(active_obj - full_obj) <= TOL_OBJ, (
        f"Objective mismatch: active={active_obj:.12g}, full={full_obj:.12g}"
    )
    assert float(np.max(np.abs(active_x - full_x))) <= TOL_X, (
        "Solution mismatch: max difference is "
        f"{np.max(np.abs(active_x - full_x)):.3e}"
    )

    assert_full_kkt(
        x=full_x,
        q=q,
        c=c,
        m_matrix=m_matrix,
        row_dual=row_dual,
        col_dual=col_dual,
        lower=lower,
        upper=upper,
    )

    return active_x, active_obj


def test_known_solution_with_z() -> None:
    """Test a three-variable [w1, w2, z] problem with known optimum.

    min  w1^2 + 2 w2^2 + z
    s.t. w1 + w2     = 1
         w1      - z = 0.25
         w1, w2, z >= 0

    Substitution gives the unique optimum (0.5, 0.5, 0.25), objective 1.
    The LP initialisation sets z=0, so this test also checks that the reduced
    cost logic correctly adds z to the active set later.
    """

    q = sparse.diags([2.0, 4.0, 0.0], format="csr")
    c = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    m_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    rhs = np.array([1.0, 0.25], dtype=np.float64)
    lower = np.zeros(3, dtype=np.float64)
    upper = np.full(3, highspy.kHighsInf, dtype=np.float64)

    x, obj = compare_active_with_full(
        q, c, m_matrix, rhs, lower, upper
    )

    expected_x = np.array([0.5, 0.5, 0.25], dtype=np.float64)
    assert float(np.max(np.abs(x - expected_x))) <= TOL_X
    assert abs(obj - 1.0) <= TOL_OBJ


def test_known_upper_bound_solution() -> None:
    """Check that a variable can finish at its upper bound."""

    q = sparse.eye(2, format="csr")
    c = np.array([-2.0, 0.0], dtype=np.float64)
    m_matrix = np.array([[1.0, 1.0]], dtype=np.float64)
    rhs = np.array([1.0], dtype=np.float64)
    lower = np.zeros(2, dtype=np.float64)
    upper = np.array([0.4, 1.0], dtype=np.float64)

    x, obj = compare_active_with_full(
        q, c, m_matrix, rhs, lower, upper
    )

    expected_x = np.array([0.4, 0.6], dtype=np.float64)
    expected_obj = -0.54
    assert float(np.max(np.abs(x - expected_x))) <= TOL_X
    assert abs(obj - expected_obj) <= TOL_OBJ


def test_random_strictly_convex_qps(number_of_tests: int = 20) -> None:
    """Compare with full HiGHS on reproducible random convex QPs."""

    for seed in range(number_of_tests):
        rng = np.random.default_rng(seed)
        dimension = 8
        number_of_constraints = 2

        factor = rng.normal(size=(dimension, dimension))
        q_dense = factor.T @ factor + 0.5 * np.eye(dimension)
        q = sparse.csr_matrix(q_dense)
        c = rng.normal(size=dimension).astype(np.float64)

        # Positive entries make a bounded nonnegative feasible set easy to
        # construct.  The generated x_feasible proves feasibility.
        m_matrix = rng.uniform(
            0.2,
            1.2,
            size=(number_of_constraints, dimension),
        ).astype(np.float64)
        x_feasible = rng.uniform(0.1, 0.8, size=dimension)
        rhs = (m_matrix @ x_feasible).astype(np.float64)

        lower = np.zeros(dimension, dtype=np.float64)
        upper = np.ones(dimension, dtype=np.float64)

        try:
            compare_active_with_full(
                q, c, m_matrix, rhs, lower, upper
            )
        except Exception as exc:
            raise AssertionError(f"Random QP failed for seed {seed}") from exc


def main() -> None:
    tests = [
        ("known [w,z] solution", test_known_solution_with_z),
        ("known upper-bound solution", test_known_upper_bound_solution),
        ("20 random convex QPs", test_random_strictly_convex_qps),
    ]

    for name, test in tests:
        test()
        print(f"PASS: {name}")

    print("\nAll qp_solver verification tests passed.")


if __name__ == "__main__":
    main()
