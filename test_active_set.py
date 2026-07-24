import numpy as np
from scipy import sparse

from robustocs.solvers1 import highs_standard_genetics


dimension = 4

sires = [0, 1]
dams = [2, 3]

mu = np.array([1.0, 0.8, 0.9, 0.7], dtype=np.float64)

sigma = sparse.csr_matrix(np.array([
    [1.0, 0.1, 0.0, 0.0],
    [0.1, 1.2, 0.0, 0.0],
    [0.0, 0.0, 1.1, 0.2],
    [0.0, 0.0, 0.2, 1.3],
], dtype=np.float64))

lam = 0.5

solution, objective_value = highs_standard_genetics(
    sigma=sigma,
    mu=mu,
    sires=sires,
    dams=dams,
    lam=lam,
    dimension=dimension,
    upper_bound=1.0,
    lower_bound=0.0,
    debug=True,
)

print("solution =", solution)
print("objective_value =", objective_value)
print("sire sum =", solution[sires].sum())
print("dam sum =", solution[dams].sum())

assert np.all(solution >= -1e-7)
assert np.all(solution <= 1.0 + 1e-7)
assert abs(solution[sires].sum() - 0.5) < 1e-7
assert abs(solution[dams].sum() - 0.5) < 1e-7
assert np.isfinite(objective_value)

print("PASS")