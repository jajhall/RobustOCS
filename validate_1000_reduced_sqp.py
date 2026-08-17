#!/usr/bin/env python3
"""Validate the new reduced-space active-set SQP on the 1000-candidate case.

Dissertation mapping used here:
    A1000.txt -> Sigma
    S1000.txt -> Omega

Run from the RobustOCS repository root:
    python validate_1000_reduced_sqp.py --tol 1e-6
Then, if that passes:
    python validate_1000_reduced_sqp.py --tol 1e-7
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

import robustocs as rocs
import robustocs.qp_solver as qp_module
from robustocs.solvers_restructured import highs_robust_genetics_sqp

LAM = 0.5
KAPPA = 1.0
RC_TOL = 1e-8
ACTIVE_TOL = 1e-10
SEX_TOL = 1e-8
BOUND_TOL = 1e-9


def symmetric_for_matvec(matrix):
    if sparse.issparse(matrix):
        m = matrix.tocsr().astype(np.float64)
    else:
        m = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    d = m - m.T
    d.eliminate_zeros()
    if d.nnz == 0:
        return m
    return (m + m.T - sparse.diags(m.diagonal(), format="csr")).tocsr()


class QPTracker:
    """Record dimensions of LP/QP subproblems passed to HiGHS."""
    def __init__(self):
        self.records = []
        self.original = None

    def __enter__(self):
        self.original = qp_module._solve_reduced_highs_qp

        def wrapped(*args, **kwargs):
            active_indices = kwargs.get("active_indices")
            use_quadratic = kwargs.get("use_quadratic")
            if active_indices is None and len(args) > 7:
                active_indices = args[7]
            if use_quadratic is None and len(args) > 8:
                use_quadratic = args[8]
            active_indices = np.asarray(active_indices, dtype=int)
            rec = {
                "type": "QP" if use_quadratic else "LP",
                "dimension": int(active_indices.size),
                "success": False,
            }
            try:
                out = self.original(*args, **kwargs)
                rec["success"] = True
                return out
            finally:
                self.records.append(rec)

        qp_module._solve_reduced_highs_qp = wrapped
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        qp_module._solve_reduced_highs_qp = self.original


def load_1000():
    d = Path("examples") / "1000"
    needed = [d/"S1000.txt", d/"A1000.txt", d/"EBV1000.txt", d/"SEX1000.txt"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Run this script from the RobustOCS repository root. Missing:\n" + "\n".join(missing)
        )
    return rocs.load_problem(
        sigma_filename=str(d / "A1000.txt"),
        mu_filename=str(d / "EBV1000.txt"),
        omega_filename=str(d / "S1000.txt"),
        sex_filename=str(d / "SEX1000.txt"),
        issparse=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    sigma, mubar, omega, n, sires, dams, names = load_1000()
    if n != 1000:
        raise RuntimeError(f"Expected n=1000, got n={n}")

    print("=" * 82)
    print("1000-CANDIDATE REDUCED-SPACE ACTIVE-SET SQP VALIDATION")
    print("Mapping: A1000 -> Sigma, S1000 -> Omega")
    print(f"robust-gap tolerance = {args.tol:.1e}")
    print("=" * 82)

    start = perf_counter()
    with QPTracker() as tracker:
        w, z, returned_obj = highs_robust_genetics_sqp(
            sigma=sigma,
            mubar=mubar,
            omega=omega,
            sires=sires,
            dams=dams,
            lam=LAM,
            kappa=KAPPA,
            dimension=n,
            lower_bound=0.0,
            upper_bound=1.0,
            max_iterations=1000,
            robust_gap_tol=args.tol,
            qp_max_iterations=1000,
            reduced_cost_tol=RC_TOL,
            active_tol=ACTIVE_TOL,
            debug=args.debug,
        )
    runtime = perf_counter() - start

    w = np.asarray(w, dtype=np.float64)
    mubar = np.asarray(mubar, dtype=np.float64)
    sigma_full = symmetric_for_matvec(sigma)
    omega_full = symmetric_for_matvec(omega)

    merit = float(mubar @ w)
    coancestry = float(w @ (sigma_full @ w))
    qform = float(w @ (omega_full @ w))
    alpha = sqrt(max(0.0, qform))
    gap = abs(float(z) - alpha)

    objective_z = merit - 0.5 * LAM * coancestry - KAPPA * float(z)
    objective_exact = merit - 0.5 * LAM * coancestry - KAPPA * alpha
    sire_sum = float(np.sum(w[list(sires)]))
    dam_sum = float(np.sum(w[list(dams)]))

    qp_dims = [r["dimension"] for r in tracker.records if r["type"] == "QP"]
    lp_count = sum(r["type"] == "LP" for r in tracker.records)
    failed_qps = sum(r["type"] == "QP" and not r["success"] for r in tracker.records)
    full_qps = sum(r["type"] == "QP" and r["dimension"] == 1001 for r in tracker.records)

    checks = {
        "finite solution": bool(np.all(np.isfinite(w)) and np.isfinite(z) and np.isfinite(returned_obj)),
        "lower bounds": float(np.min(w)) >= -BOUND_TOL,
        "upper bounds": float(np.max(w)) <= 1.0 + BOUND_TOL,
        "sire sum = 0.5": abs(sire_sum - 0.5) <= SEX_TOL,
        "dam sum = 0.5": abs(dam_sum - 0.5) <= SEX_TOL,
        "robust gap": gap <= args.tol,
        "returned objective consistency": abs(float(returned_obj) - objective_z) <= 5e-7,
        "exact objective consistency": abs(float(returned_obj) - objective_exact) <= args.tol + 5e-7,
        "reduced-space actually used": bool(qp_dims and min(qp_dims) < 1001),
    }

    print("\nSOLUTION")
    print(f"runtime (s)                    = {runtime:.6f}")
    print(f"returned objective             = {returned_obj:.12e}")
    print(f"objective recomputed using z   = {objective_z:.12e}")
    print(f"exact robust objective         = {objective_exact:.12e}")
    print(f"z                              = {float(z):.12e}")
    print(f"sqrt(w^T Omega w)              = {alpha:.12e}")
    print(f"absolute robust gap            = {gap:.3e}")
    print(f"sire contribution sum          = {sire_sum:.12e}")
    print(f"dam contribution sum           = {dam_sum:.12e}")
    print(f"nonzero contributions (>1e-10)= {int(np.sum(w > ACTIVE_TOL))}")
    print(f"minimum w                      = {float(np.min(w)):.3e}")
    print(f"maximum w                      = {float(np.max(w)):.3e}")

    print("\nREDUCED-SPACE BEHAVIOUR")
    print("full QP dimension              = 1001")
    print(f"number of full-variable LPs    = {lp_count}")
    print(f"number of QP attempts          = {len(qp_dims)}")
    if qp_dims:
        print(f"minimum QP dimension           = {min(qp_dims)}")
        print(f"median QP dimension            = {float(np.median(qp_dims)):.1f}")
        print(f"mean QP dimension              = {float(np.mean(qp_dims)):.3f}")
        print(f"maximum QP dimension           = {max(qp_dims)}")
    print(f"full-dimension QP attempts     = {full_qps}")
    print(f"failed QP attempts             = {failed_qps}")

    print("\nVALIDATION CHECKS")
    for name, ok in checks.items():
        print(f"{name:<38}: {'PASS' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nPASS: 1000-candidate reduced-space active-set SQP validation succeeded.")
    else:
        failed = [name for name, ok in checks.items() if not ok]
        print("\nFAIL: validation failed for: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
