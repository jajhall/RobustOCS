#!/usr/bin/env python3

import numpy as np
import robustocs as rocs

np.set_printoptions(
    precision=10,
    suppress=True,
    linewidth=160
)

sigma, mubar, omega, n, _, _, _ = rocs.load_problem(
    sigma_filename="A50.txt",
    mu_filename="EBV50.txt",
    omega_filename="S50.txt",
    issparse=True
)

sires = range(0, n, 2)
dams = range(1, n, 2)

lam = 0.5

w, obj = rocs.highs_standard_genetics(
    sigma,
    mubar,
    sires,
    dams,
    lam,
    n
)

print("=" * 80)
print("HiGHS standard genetics solution")
print("=" * 80)

print("\nOptimal contribution vector w:")
for i, value in enumerate(w):
    sex = "sire" if i % 2 == 0 else "dam"
    print(f"Individual {i:2d} ({sex}): {value:.10f}")

print("\nObjective value:")
print(f"{obj:.10f}")

print("\nConstraint checks:")
print(f"Sum of sire contributions = {np.sum(w[list(sires)]):.10f}")
print(f"Sum of dam contributions  = {np.sum(w[list(dams)]):.10f}")
print(f"Minimum contribution      = {np.min(w):.10f}")
print(f"Maximum contribution      = {np.max(w):.10f}")
print(f"Number of nonzero entries = {np.sum(np.abs(w) > 1e-8)}")