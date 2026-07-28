#!/usr/bin/env python3
"""Verify Robust OCS SQP using the user's general QP solver.

This script performs two checks:

1. The three-candidate example in the Robust OCS paper.
2. The real 50-candidate data, using the requested mapping:
       S50.txt   -> sigma
       A50.txt   -> omega
       EBV50.txt -> mubar
       SEX50.txt -> sire/dam sets

Run from the RobustOCS repository root:

    python test_robust_ocs_qp_solver.py
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
from scipy import sparse

import robustocs as rocs
from robustocs.solvers_restructured import highs_robust_genetics_sqp


def full_symmetric(
    matrix: np.ndarray | sparse.spmatrix
) -> sparse.csr_matrix:
    """Return a complete symmetric CSR matrix."""

    if sparse.issparse(matrix):
        result = matrix.tocsr().astype(np.float64)
    else:
        result = sparse.csr_matrix(
            np.asarray(matrix, dtype=np.float64)
        )

    difference = result - result.transpose()
    difference.eliminate_zeros()

    if difference.nnz == 0:
        return result

    return (
        result
        + result.transpose()
        - sparse.diags(result.diagonal(), format="csr")
    ).tocsr()


def robust_objective(
    w: np.ndarray,
    mubar: np.ndarray,
    sigma: np.ndarray | sparse.spmatrix,
    omega: np.ndarray | sparse.spmatrix,
    lam: float,
    kappa: float,
) -> float:
    """Evaluate the original Robust OCS objective, equation (8)."""

    sigma_full = full_symmetric(sigma)
    omega_full = full_symmetric(omega)

    coancestry = float(w @ (sigma_full @ w))
    uncertainty = sqrt(
        max(0.0, float(w @ (omega_full @ w)))
    )

    return float(
        mubar @ w
        - 0.5 * lam * coancestry
        - kappa * uncertainty
    )


def test_paper_example() -> None:
    """Reproduce the three-candidate example from the paper."""

    mubar = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    omega = sparse.diags(
        [1.0 / 9.0, 4.0, 1.0],
        format="csr",
        dtype=np.float64,
    )
    sigma = sparse.identity(3, format="csr", dtype=np.float64)

    sires = [0, 1]
    dams = [2]
    lam = 0.1
    kappa = 1.0

    w, z, objective = highs_robust_genetics_sqp(
        sigma=sigma,
        mubar=mubar,
        omega=omega,
        sires=sires,
        dams=dams,
        lam=lam,
        kappa=kappa,
        dimension=3,
        lower_bound=0.0,
        upper_bound=1.0,
        max_iterations=200,
        robust_gap_tol=1e-7,
        qp_max_iterations=200,
        reduced_cost_tol=1e-8,
        active_tol=1e-10,
        debug=False,
    )

    expected_w = np.array(
        [0.3359, 0.1641, 0.5],
        dtype=np.float64,
    )
    expected_objective = 0.5361

    true_uncertainty = sqrt(float(w @ (omega @ w)))
    objective_from_original_problem = robust_objective(
        w=w,
        mubar=mubar,
        sigma=sigma,
        omega=omega,
        lam=lam,
        kappa=kappa,
    )

    print("=" * 80)
    print("Paper three-candidate example")
    print("=" * 80)
    print("w =", np.array2string(w, precision=10))
    print(f"z = {z:.12e}")
    print(f"sqrt(w^T Omega w) = {true_uncertainty:.12e}")
    print(f"returned objective = {objective:.12e}")
    print(
        "objective from equation (8) = "
        f"{objective_from_original_problem:.12e}"
    )

    assert np.max(np.abs(w - expected_w)) <= 5e-4, (
        "The computed contribution vector does not reproduce the "
        "paper's example."
    )
    assert abs(objective_from_original_problem - expected_objective) <= 5e-4
    assert abs(z - true_uncertainty) <= 1e-7
    assert abs(np.sum(w[sires]) - 0.5) <= 1e-8
    assert abs(np.sum(w[dams]) - 0.5) <= 1e-8
    assert abs(objective - objective_from_original_problem) <= 1e-7

    print("PASS: paper example reproduced.\n")


def test_real_50() -> None:
    """Solve the requested real 50-dimensional Robust OCS instance."""

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
            "Run this script from the RobustOCS repository root. "
            "The following data files were not found:\n"
            + "\n".join(missing)
        )

    # Mapping specified by the supervisor:
    # S50 is Sigma and A50 is Omega.
    sigma, mubar, omega, dimension, sires, dams, names = (
        rocs.load_problem(
            sigma_filename=str(data_dir / "S50.txt"),
            mu_filename=str(data_dir / "EBV50.txt"),
            omega_filename=str(data_dir / "A50.txt"),
            sex_filename=str(data_dir / "SEX50.txt"),
            issparse=True,
        )
    )

    lam = 0.5
    kappa = 1.0

    print("=" * 80)
    print("Real 50-dimensional Robust OCS")
    print("Mapping: S50 -> Sigma, A50 -> Omega")
    print("=" * 80)

    w, z, objective = highs_robust_genetics_sqp(
        sigma=sigma,
        mubar=mubar,
        omega=omega,
        sires=sires,
        dams=dams,
        lam=lam,
        kappa=kappa,
        dimension=dimension,
        lower_bound=0.0,
        upper_bound=1.0,
        max_iterations=1000,
        robust_gap_tol=1e-7,
        qp_max_iterations=1000,
        reduced_cost_tol=1e-8,
        active_tol=1e-10,
        debug=True,
    )

    sigma_full = full_symmetric(sigma)
    omega_full = full_symmetric(omega)

    sire_indices = list(sires)
    dam_indices = list(dams)

    sire_total = float(np.sum(w[sire_indices]))
    dam_total = float(np.sum(w[dam_indices]))
    uncertainty = sqrt(
        max(0.0, float(w @ (omega_full @ w)))
    )
    coancestry = float(w @ (sigma_full @ w))

    objective_from_original_problem = robust_objective(
        w=w,
        mubar=np.asarray(mubar, dtype=np.float64),
        sigma=sigma,
        omega=omega,
        lam=lam,
        kappa=kappa,
    )

    print("\n" + "=" * 80)
    print("Final verification")
    print("=" * 80)
    print(f"dimension = {dimension}")
    print(f"objective returned = {objective:.12e}")
    print(
        "objective from equation (8) = "
        f"{objective_from_original_problem:.12e}"
    )
    print(f"genetic merit = {float(mubar @ w):.12e}")
    print(f"coancestry w^T Sigma w = {coancestry:.12e}")
    print(f"sire contribution sum = {sire_total:.12e}")
    print(f"dam contribution sum = {dam_total:.12e}")
    print(f"z = {z:.12e}")
    print(f"sqrt(w^T Omega w) = {uncertainty:.12e}")
    print(f"z - sqrt(w^T Omega w) = {z - uncertainty:.3e}")
    print(f"nonzero contributions = {int(np.sum(w > 1e-10))}")
    print(f"minimum w = {float(np.min(w)):.3e}")
    print(f"maximum w = {float(np.max(w)):.3e}")

    order = np.argsort(-w)
    print("\nTen largest contributions:")
    for index in order[:10]:
        name = names[index] if names is not None else str(index)
        print(
            f"  index={index:2d}, name={name}, "
            f"w={w[index]:.12e}"
        )

    assert abs(sire_total - 0.5) <= 1e-7
    assert abs(dam_total - 0.5) <= 1e-7
    assert float(np.min(w)) >= -1e-9
    assert abs(z - uncertainty) <= 5e-7
    assert abs(objective - objective_from_original_problem) <= 5e-7

    print("\nPASS: real 50-dimensional Robust OCS checks passed.")


def main() -> None:
    test_paper_example()
    test_real_50()
    print("\nAll Robust OCS tests passed.")


if __name__ == "__main__":
    main()
