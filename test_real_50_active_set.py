import numpy as np
import robustocs as rocs

from robustocs.solvers2 import highs_standard_genetics


sigma, mu, omega, n, sires, dams, names = rocs.load_problem(
    sigma_filename="examples/50/A50.txt",
    mu_filename="examples/50/EBV50.txt",
    omega_filename="examples/50/S50.txt",
    sex_filename="examples/50/SEX50.txt",
    issparse=True
)

print("=" * 80)
print("Real package-provided 50-dimensional example")
print("=" * 80)

print("dimension =", n)
print("number of sires =", len(sires))
print("number of dams =", len(dams))
print("sigma shape =", sigma.shape)
print("omega shape =", omega.shape)

solution, objective_value = highs_standard_genetics(
    sigma=sigma,
    mu=mu,
    sires=sires,
    dams=dams,
    lam=0.5,
    dimension=n,
    upper_bound=1.0,
    lower_bound=0.0,
    debug=True
)

print("\nFinal result")
print("solution =", solution)
print("objective_value =", objective_value)
print("sire sum =", solution[sires].sum())
print("dam sum =", solution[dams].sum())
print("number of nonzero weights =", np.sum(solution > 1e-8))

true_std = np.loadtxt("examples/50/solution_std.txt")

max_difference = np.max(np.abs(solution - true_std))

print("\nComparison with provided standard solution")
print("max difference from provided standard solution =", max_difference)

assert abs(solution[sires].sum() - 0.5) < 1e-7
assert abs(solution[dams].sum() - 0.5) < 1e-7
assert np.all(solution >= -1e-7)
assert np.all(solution <= 1.0 + 1e-7)
assert max_difference < 1e-3

print("\nPASS")