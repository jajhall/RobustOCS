# -*- coding: utf-8 -*-
"""Defining Solvers

With an optimal contribution selection problems properly loaded into Python,
`solvers` contains functions for solving those under various formulations and
methods.

Documentation is available in the docstrings and online at
https://github.com/Foggalong/RobustOCS/wiki
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
import gurobipy as gp       # Gurobi optimization interface
import highspy              # HiGHS optimization interface
from math import sqrt       # used within the robust constraint
from scipy import sparse    # used for sparse matrix format
from time import time       # used for timing SQP methods

# controls what's imported on `from robustocs.solvers import *`
__all__ = [
    "gurobi_standard_genetics",
    "gurobi_robust_genetics",  # alias
    "gurobi_robust_genetics_conic",
    "gurobi_robust_genetics_sqp",
    "highs_standard_genetics",
    "highs_robust_genetics",  # alias
    "highs_robust_genetics_sqp"
]


def gurobi_standard_genetics(
    sigma: npt.NDArray[np.float64] | sparse.spmatrix,
    mu: npt.NDArray[np.float64],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Solve the standard genetic selection problem using Gurobi.

    Given a standard genetic selection problem
    ```
        max_w w'mu - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    this function uses Gurobi to find the optimum w and the objective for that
    portfolio. Additional parameters give control over long Gurobi can spend
    on the problem, to prevent indefinite hangs.

    Parameters
    ----------
    sigma : ndarray or spmatrix
        Covariance matrix of the candidates in the cohorts for selection.
    mu : ndarray
        Vector of expected returns for candidates in the cohorts for selection.
    sires : Any
        An object representing an index set for sires (male candidates) in the
        cohort. Type is not restricted.
    dams : Any
        An object representing an index set for dams (female candidates) in the
        cohort. Type is not restricted.
    lam : float
        Lambda value to optimize for, which controls the balance between risk
        and return. Lower values will give riskier portfolios, higher values
        more conservative ones.
    dimension : int
        Number of candidates in the cohort, i.e. the dimension of the problem.
    upper_bound : ndarray or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give Gurobi to solve the problem.
        Default value is `None`, i.e. no time limit.
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # create models for standard and robust genetic selection
    model = gp.Model("standard-genetics")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        model.setParam('OutputFlag', 0)

    # integrating bounds within variable definitions is more efficient than
    # as a separate constraint, which Gurobi would convert to bounds anyway
    w = model.addMVar(shape=dimension, lb=lower_bound, ub=upper_bound,
                      vtype=gp.GRB.CONTINUOUS, name="w")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mu - (lam/2)*w.transpose()@(sigma@w),
        gp.GRB.MAXIMIZE
    )

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension), dtype=bool)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5)
    model.addConstr(M@w == m, name="sum-to-half")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)

    # model file can be used externally for verification
    if model_output:
        model.write(f"{model_output}.mps")

    model.optimize()
    return np.array(w.X), model.ObjVal  # HACK np.array avoids issue #9


def gurobi_robust_genetics_conic(
    sigma: npt.NDArray[np.float64] | sparse.spmatrix,
    mubar: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64] | sparse.spmatrix,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Solve the robust genetic selection problem using Gurobi.

    Given a robust genetic selection problem
    ```
        max_w (min_mu w'mu subject to mu in U) - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    where U is a quadratic uncertainty set for mu~N(mubar, omega), this
    function uses Gurobi to find the optimum w and the objective for that
    portfolio. It first uses the KKT conditions to find exactly the solution
    to the inner problem, substitutes that into the outer problem, and then
    relaxes the uncertainty term into a constraint before solving.

    Additional parameters give control over long Gurobi can spend
    on the problem, to prevent indefinite hangs.

    Parameters
    ----------
    sigma : ndarray or spmatrix
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    omega : ndarray or spmatrix
        Covariance matrix for expected returns for candidates in the cohort for
        selection.
    sires : Any
        An object representing an index set for sires (male candidates) in the
        cohort. Type is not restricted.
    dams : Any
        An object representing an index set for dams (female candidates) in the
        cohort. Type is not restricted.
    lam : float
        Lambda value to optimize for, which controls the balance between risk
        and return. Lower values will give riskier portfolios, higher values
        more conservative ones.
    kappa : float
        Kappa value to optimize for, which controls how resilient the solution
        must be to variation in expected values.
    dimension : int
        Number of candidates in the cohort, i.e. the dimension of the problem.
    upper_bound : ndarray or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give Gurobi to solve the problem.
        Default value is `None`, i.e. no time limit.
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Auxiliary variable corresponding to uncertainty associated with the
        portfolio vector which Gurobi has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # create models for standard and robust genetic selection
    model = gp.Model("robust-genetics")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        model.setParam('OutputFlag', 0)

    # integrating bounds within variable definitions is more efficient than
    # as a separate constraint, which Gurobi would convert to bounds anyway
    w = model.addMVar(shape=dimension, lb=lower_bound, ub=upper_bound,
                      vtype=gp.GRB.CONTINUOUS, name="w")
    z = model.addVar(lb=0.0, name="z")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mubar - (lam/2)*w.transpose()@(sigma@w) - kappa*z,
        gp.GRB.MAXIMIZE
    )

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension), dtype=bool)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5, dtype=np.float64)
    model.addConstr(M@w == m, name="sum-to-half")

    # conic constraint which comes from robust optimization
    model.addConstr(z**2 >= w.transpose()@omega@w, name="uncertainty")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)

    # model file can be used externally for verification
    if model_output:
        model.write(f"{model_output}.mps")

    model.optimize()
    return np.array(w.X), z.X, model.ObjVal  # HACK np.array avoids issue #9


def gurobi_robust_genetics_sqp(
    sigma: npt.NDArray[np.float64] | sparse.spmatrix,
    mubar: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64] | sparse.spmatrix,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    robust_gap_tol: float = 1e-7,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Solve the robust genetic selection problem using SQP in Gurobi.

    Given a robust genetic selection problem
    ```
        max_w (min_mu w'mu subject to mu in U) - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    where U is a quadratic uncertainty set for mu~N(mubar, omega), this
    function uses Gurobi to find the optimum w and the objective for that
    portfolio. It does this using sequential quadratic programming (SQP),
    approximating the conic constraint associated with robustness using
    a series of linear constraints.

    Additional parameters give control over long Gurobi can spend
    on the problem, to prevent indefinite hangs.

    Parameters
    ----------
    sigma : ndarray or spmatrix
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    omega : ndarray or spmatrix
        Covariance matrix for expected returns for candidates in the cohort for
        selection.
    sires : Any
        An object representing an index set for sires (male candidates) in the
        cohort. Type is not restricted.
    dams : Any
        An object representing an index set for dams (female candidates) in the
        cohort. Type is not restricted.
    lam : float
        Lambda value to optimize for, which controls the balance between risk
        and return. Lower values will give riskier portfolios, higher values
        more conservative ones.
    kappa : float
        Kappa value to optimize for, which controls how resilient the solution
        must be to variation in expected values.
    dimension : int
        Number of candidates in the cohort, i.e. the dimension of the problem.
    upper_bound : ndarray or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give Gurobi to solve the problem.
        Default value is `None`, i.e. no time limit.
    max_iterations : int, optional
        Maximum number of iterations that can be taken in solving the problem,
        i.e. the maximum number of constraints to use to approximate the conic
        constraint. Default value is `1000`.
    robust_gap_tol : float, optional
        Tolerance when checking whether an approximating constraint is active
        and whether the SQP overall has converged. Default value is 10^-7.
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Auxiliary variable corresponding to uncertainty associated with the
        portfolio vector which Gurobi has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # create models for standard and robust genetic selection
    model = gp.Model("robust-genetics-sqp")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        model.setParam('OutputFlag', 0)

    # integrating bounds within variable definitions is more efficient than
    # as a separate constraint, which Gurobi would convert to bounds anyway
    w = model.addMVar(shape=dimension, lb=lower_bound, ub=upper_bound,
                      vtype=gp.GRB.CONTINUOUS, name="w")
    z = model.addVar(lb=0.0, name="z")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mubar - (lam/2)*w.transpose()@(sigma@w) - kappa*z,
        gp.GRB.MAXIMIZE
    )

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension), dtype=bool)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5, dtype=np.float64)
    model.addConstr(M@w == m, name="sum-to-half")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        time_remaining: float = time_limit

    for i in range(max_iterations):
        # optional controls to stop Gurobi taking too long
        if time_limit:
            model.setParam(gp.GRB.Param.TimeLimit, time_remaining)

        # optimization of the model, print weights and objective
        try:
            model.optimize()
        except RuntimeError as e:
            raise RuntimeError(f"Gurobi failed with error:\n{e}")

        # subtract time take from time remaining
        if time_limit:
            time_remaining -= model.Runtime
            if time_remaining < 0:
                raise RuntimeError(f"Gurobi hit time limit of {time_limit} "
                                   f"seconds without reaching optimality")

        # return model and solution at every approximation to help debug
        if model_output:
            model.write(f"{model_output}.mps")
        if debug:
            print(f"{i}: {w.X}, {model.ObjVal:g}")

        # assess which constraints are currently active
        active_const: bool = False
        for c in model.getConstrs():
            if abs(c.Slack) > robust_gap_tol:
                active_const = True
                if debug:
                    print(f"{c.ConstrName} active, slack {c.Slack:g}")
        if debug and not active_const:
            print("No active constraints!")

        # z coefficient for the new constraint
        w_star: npt.NDArray[np.float64] = np.array(w.X)
        alpha: float = sqrt(w_star.transpose()@omega@w_star)

        # if gap between z and w'Omega w has converged, done
        if abs(z.X - alpha) < robust_gap_tol:
            break

        # add a new plane to the approximation of the uncertainty cone
        model.addConstr(alpha*z >= w_star.transpose()@omega@w, name=f"P{i}")

    return np.array(w.X), z.X, model.ObjVal  # HACK np.array avoids issue #9


# make gurobi_robust_genetics(...) an alias of the fastest method
gurobi_robust_genetics = gurobi_robust_genetics_conic


def highs_bound_like(dimension: int,
                     value: float | list[float] | npt.NDArray[np.float64]
                     ):  # BUG broke: npt.NDArray[np.float64] | list[float]
    """
    Helper function which allows HiGHS to interpret variable bounds specified
    either as a vector or a single floating point value. If `value` is an array
    will just return that array. If `value` is a float, it'll return a NumPy
    array in the shape of `vector` with every entry being `value`.
    """

    return [value]*dimension if type(value) is float else value


def _highs_bound_array(
    dimension: int,
    value: float | list[float] | npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Convert scalar/list/array bounds into a NumPy array of length dimension.
    This is needed because the active-set method modifies bounds directly.
    """

    if np.isscalar(value):
        return np.full(dimension, float(value), dtype=np.float64)

    value_array = np.asarray(value, dtype=np.float64)

    if value_array.shape != (dimension,):
        raise ValueError(
            f"Bound must be scalar or shape ({dimension},), "
            f"got shape {value_array.shape}"
        )

    return value_array.copy()


def _remaining_time(
    time_limit: float | None,
    start_time: float | None
) -> float | None:
    """
    Compute remaining time for the active-set algorithm.
    """

    if time_limit is None:
        return None

    if start_time is None:
        return time_limit

    remaining = time_limit - (time() - start_time)

    if remaining <= 0:
        raise RuntimeError(
            f"HiGHS hit time limit of {time_limit} seconds "
            f"without reaching optimality"
        )

    return remaining


def _solve_highs_standard_subproblem(
    sigma: sparse.spmatrix,
    mu: npt.NDArray[np.float64],
    sires,
    dams,
    lam: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | list[float] | float,
    lower_bound: npt.NDArray[np.float64] | list[float] | float,
    active_set: npt.NDArray[np.bool_] | None,
    use_quadratic: bool,
    time_limit: float | None,
    model_output: str,
    debug: bool,
    iteration: int
) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
    """
    Solve either the initial LP or one restricted QP.

    If active_set is None:
        solve over all variables.

    If active_set is not None:
        variables outside active_set are fixed to zero.
    """

    # initialise an empty model
    h = highspy.Highs()
    model = highspy.HighsModel()

    lower = _highs_bound_array(dimension, lower_bound)
    upper = _highs_bound_array(dimension, upper_bound)

    # Restrict the model to the current active set N_h.
    # Inactive variables are fixed at w_i = 0.
    if active_set is not None:
        lower[~active_set] = 0.0
        upper[~active_set] = 0.0

    # NOTE HiGHS doesn't support typing for model parameters
    model.lp_.model_name_ = "standard-genetics-active-set"
    model.lp_.num_col_ = dimension
    model.lp_.num_row_ = 2

    # HiGHS does minimization, so negate objective.
    model.lp_.col_cost_ = -mu

    # bounds on w
    model.lp_.col_lower_ = lower
    model.lp_.col_upper_ = upper

    # For the initial LP, do not include the quadratic term.
    # For restricted QP iterations, include it.
    if use_quadratic:
        model.hessian_.format_ = highspy.HessianFormat.kSquare
        model.hessian_.dim_ = dimension
        model.hessian_.start_ = sigma.indptr
        model.hessian_.index_ = sigma.indices

        # HiGHS multiplies Hessian by 1/2, so just need factor of lambda.
        model.hessian_.value_ = lam*sigma.data

    # add Mx = m to the model using CSR format
    model.lp_.row_lower_ = model.lp_.row_upper_ = np.full(2, 0.5)
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = [0, len(sires), dimension]
    model.lp_.a_matrix_.index_ = list(sires) + list(dams)
    model.lp_.a_matrix_.value_ = [1]*dimension

    # HiGHS spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        h.setOptionValue('output_flag', False)
        h.setOptionValue('log_to_console', False)

    # HiGHS' passModel returns a status indicating its success
    pass_status: highspy._core.HighsStatus = h.passModel(model)

    # model file must be saved between passModel and any error
    if model_output:
        h.writeModel(f"{model_output}_iter_{iteration}.mps")

    # HiGHS will try to continue if it gets an error, so stop it
    if pass_status == highspy.HighsStatus.kError:
        raise ValueError(
            f"h.passModel failed with status {h.getModelStatus()}"
        )

    # optional controls to stop HiGHS taking too long
    if time_limit:
        h.setOptionValue('time_limit', h.getRunTime() + time_limit)

    # HiGHS' run returns a status indicating its success
    run_status: highspy._core.HighsStatus = h.run()

    # solution with dual info must be printed between run and any error
    if debug:
        h.writeSolution("", 1)

    model_status: highspy._core.HighsModelStatus = h.getModelStatus()

    # HiGHS will try to continue if it gets an error, so stop it
    if run_status == highspy.HighsStatus.kError:
        raise ValueError(f"h.run failed with status {model_status}")
    elif model_status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(
            f"h.run did not achieve optimality, status {model_status}"
        )

    highs_solution = h.getSolution()

    # by default, col_value is a stock-Python list
    solution: npt.NDArray[np.float64] = np.array(highs_solution.col_value)

    # we negated the objective function, so negate it back
    objective_value: float = -h.getInfo().objective_function_value

    # row_dual contains the dual variables for the two constraints Mx = m
    row_dual: npt.NDArray[np.float64] = np.array(highs_solution.row_dual)

    return solution, objective_value, row_dual


def highs_standard_genetics(
    sigma: sparse.spmatrix,
    mu: npt.NDArray[np.float64],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | list[float] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | list[float] | float = 0.0,
    time_limit: float | None = None,
    model_output: str = '',
    debug: bool = False,
    max_iterations: int = 1000,
    reduced_cost_tol: float = 1e-8,
    active_tol: float = 1e-10
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Solve the standard genetic selection problem using HiGHS.

    This version implements the active-set / column-generation method from the
    handwritten notes.

    Step 0:
        Solve the LP

            max_w mu'w
            subject to M w = m,
                       0 <= w <= upper_bound.

        Let N_0 = {i : w_i > 0}.

    Step h:
        Solve the restricted QP

            max_w mu'w - lambda/2 w'Sigma w
            subject to M w = m,
                       0 <= w_i <= upper_bound_i for i in N_h,
                       w_i = 0 for i not in N_h.

        Compute reduced costs for inactive variables. If all inactive reduced
        costs are non-positive, stop. Otherwise add variables with positive
        reduced cost to N_h.
    """

    # HiGHS Hessian input expects sparse matrix indexing.
    if not sparse.isspmatrix_csr(sigma):
        sigma = sigma.tocsr()

    mu = np.asarray(mu, dtype=np.float64)
    sires = list(sires)
    dams = list(dams)

    lower = _highs_bound_array(dimension, lower_bound)
    upper = _highs_bound_array(dimension, upper_bound)

    # The handwritten method fixes inactive variables to w_i = 0.
    # Therefore lower bounds must allow zero.
    if np.any(lower > active_tol):
        raise ValueError(
            "Active-set method assumes inactive variables can be fixed to 0. "
            "Therefore lower_bound must be 0 for all variables."
        )

    start_time = time() if time_limit else None

    # ------------------------------------------------------------------
    # Step 0: solve initial LP
    # ------------------------------------------------------------------
    w_lp, _, _ = _solve_highs_standard_subproblem(
        sigma=sigma,
        mu=mu,
        sires=sires,
        dams=dams,
        lam=lam,
        dimension=dimension,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        active_set=None,
        use_quadratic=False,
        time_limit=_remaining_time(time_limit, start_time),
        model_output=model_output,
        debug=debug,
        iteration=-1
    )

    # N_0 = {i : w_i > 0}
    active_set: npt.NDArray[np.bool_] = w_lp > active_tol

    # Safety check: the LP should already give at least one active sire
    # and at least one active dam because of the sum-to-half constraints.
    # This handles tiny numerical issues.
    if not np.any(active_set[sires]):
        best_sire = sires[int(np.argmax(mu[sires]))]
        active_set[best_sire] = True

    if not np.any(active_set[dams]):
        best_dam = dams[int(np.argmax(mu[dams]))]
        active_set[best_dam] = True

    # group_of_var[i] tells which row of M variable i belongs to:
    # 0 means sire row, 1 means dam row.
    group_of_var = np.empty(dimension, dtype=int)
    group_of_var[sires] = 0
    group_of_var[dams] = 1

    final_solution: npt.NDArray[np.float64] | None = None
    final_objective_value: float | None = None

    # ------------------------------------------------------------------
    # Active-set iterations
    # ------------------------------------------------------------------
    for iteration in range(max_iterations):
        solution, objective_value, row_dual = _solve_highs_standard_subproblem(
            sigma=sigma,
            mu=mu,
            sires=sires,
            dams=dams,
            lam=lam,
            dimension=dimension,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            active_set=active_set,
            use_quadratic=True,
            time_limit=_remaining_time(time_limit, start_time),
            model_output=model_output,
            debug=debug,
            iteration=iteration
        )

        final_solution = solution
        final_objective_value = objective_value

        # Gradient of the maximization objective:
        #
        #     grad_i = mu_i - lambda * (Sigma w)_i
        #
        # Reduced cost for inactive variable i:
        #
        #     rc_i = grad_i - M_i^T y
        #
        # row_dual has two entries, one for sire sum constraint and one for
        # dam sum constraint.
        reduced_cost: npt.NDArray[np.float64] = (
            mu - lam*(sigma @ solution) + row_dual[group_of_var]
        )

        inactive_set = ~active_set

        # Candidate variables that should enter the active set.
        entering_set = (
            inactive_set
            & (upper > active_tol)
            & (reduced_cost > reduced_cost_tol)
        )

        if debug:
            if np.any(inactive_set):
                max_inactive_reduced_cost = np.max(reduced_cost[inactive_set])
            else:
                max_inactive_reduced_cost = None

            print(
                f"active-set iteration {iteration}: "
                f"active variables = {np.sum(active_set)}, "
                f"max inactive reduced cost = {max_inactive_reduced_cost}"
            )

        # Stop condition:
        # all inactive variables have non-positive reduced cost.
        if not np.any(entering_set):
            return final_solution, final_objective_value

        # Add every inactive variable with positive reduced cost.
        active_set[entering_set] = True

    raise RuntimeError(
        f"Active-set HiGHS solver did not converge after "
        f"{max_iterations} iterations"
    )


def highs_robust_genetics_sqp(
    sigma: sparse.spmatrix,
    mubar: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64] | sparse.spmatrix,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | list[float] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | list[float] | float = 0.0,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    robust_gap_tol: float = 1e-7,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Solve the robust genetic selection problem using SQP in HiGHS.

    Given a robust genetic selection problem
    ```
        max_w (min_mu w'mu subject to mu in U) - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    where U is a quadratic uncertainty set for mu~N(mubar, omega), this
    function uses HiGHS to find the optimum w and the objective for that
    portfolio. It does this using sequential quadratic programming (SQP),
    approximating the conic constraint associated with robustness using
    a series of linear constraints.

    Parameters
    ----------
    sigma : spmatrix
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    omega : ndarray or spmatrix   # TODO this doesn't *have* to be an spmatrix
        Covariance matrix for expected returns for candidates in the cohort for
        selection.
    sires : Any
        An object representing an index set for sires (male candidates) in the
        cohort. Type is not restricted.
    dams : Any
        An object representing an index set for dams (female candidates) in the
        cohort. Type is not restricted.
    lam : float
        Lambda value to optimize for, which controls the balance between risk
        and return. Lower values will give riskier portfolios, higher values
        more conservative ones.
    kappa : float
        Kappa value to optimize for, which controls how resilient the solution
        must be to variation in expected values.
    dimension : int
        Number of candidates in the cohort, i.e. the dimension of the problem.
    upper_bound : ndarray, list, or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray, list, or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give HiGHS to solve the problem.
        Default value is `None`, i.e. no time limit.
    max_iterations : int, optional
        Maximum number of iterations that can be taken in solving the problem,
        i.e. the maximum number of constraints to use to approximate the conic
        constraint. Default value is `1000`.
    robust_gap_tol : float, optional
        Tolerance when checking whether an approximating constraint is active
        and whether the SQP overall has converged. Default value is 10^-7.
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Auxiliary variable corresponding to uncertainty associated with the
        portfolio vector which Gurobi has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # initialise an empty model
    h = highspy.Highs()
    model = highspy.HighsModel()

    # use value for infinity from HiGHS
    inf = highspy.kHighsInf

    # NOTE HiGHS doesn't support typing for model parameters
    model.lp_.model_name_ = "robust-genetics"
    model.lp_.num_col_ = dimension
    model.lp_.num_row_ = 2

    # HiGHS does minimization so negate objective
    model.lp_.col_cost_ = -mubar

    # bounds on w using a helper function
    model.lp_.col_lower_ = highs_bound_like(dimension, lower_bound)
    model.lp_.col_upper_ = highs_bound_like(dimension, upper_bound)

    # define the quadratic term in the objective
    model.hessian_.format_ = highspy.HessianFormat.kSquare
    model.hessian_.dim_ = dimension
    model.hessian_.start_ = sigma.indptr
    model.hessian_.index_ = sigma.indices
    # # # HiGHS multiplies Hessian by 1/2 so just need factor of lambda
    model.hessian_.value_ = lam*sigma.data

    # add Mx = m to the model using CSR format
    model.lp_.row_lower_ = model.lp_.row_upper_ = np.full(2, 0.5)
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = [0, len(sires), dimension]
    model.lp_.a_matrix_.index_ = list(sires) + list(dams)
    model.lp_.a_matrix_.value_ = [1]*dimension

    # HiGHS' passModel returns a status indicating its success
    pass_status: highspy._core.HighsStatus = h.passModel(model)
    # model file must be saved between passModel and any error
    if model_output:
        h.writeModel(f"{model_output}.mps")
    # HiGHS will try to continue if it gets an error, so stop it
    if pass_status == highspy.HighsStatus.kError:
        raise ValueError(f"h.passModel failed with status "
                         f"{h.getModelStatus()}")

    # add z variable with bound 0 < z < inf and cost kappa
    h.addVar(0, highspy.kHighsInf)
    h.changeColCost(dimension, kappa)

    # HiGHS spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        h.setOptionValue('output_flag', False)
        h.setOptionValue('log_to_console', False)

    # optional controls to stop HiGHS taking too long
    if time_limit:
        time_remaining: float = time_limit

    for i in range(max_iterations):
        # use at most the remaining unused time (see issue #16)
        if time_limit:
            h.setOptionValue('time_limit', h.getRunTime() + time_remaining)
            start_time: float = time()

        try:
            run_status: highspy._core.HighsStatus = h.run()
        except RuntimeError as e:
            raise RuntimeError(f"HiGHS failed with error:\n{e}")

        # subtract time taken from time remaining
        if time_limit:
            time_remaining -= time() - start_time
            if time_remaining < 0:
                raise RuntimeError(f"HiGHS hit time limit of {time_limit} "
                                   f"seconds without reaching optimality")

        # return model and solution at every approximation to help debug
        if model_output:
            h.writeModel(f"{model_output}.mps")
        if debug:
            h.writeSolution("", 1)

        # evaluate HiGHS' return value from h.run and attempt to solve
        model_status: highspy._core.HighsModelStatus = h.getModelStatus()
        # HiGHS will try to continue if it gets an error, so stop it
        if run_status == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS at approximation #{i} failed with "
                               f"status {model_status}")
        elif model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"HiGHS did not achieve optimality at "
                               f"approximation #{i}, status {model_status}")

        # by default, col_value is a stock-Python list
        solution: list[float] = h.getSolution().col_value
        w_star: npt.NDArray[np.float64] = np.array(solution[:-1])
        z_star: float = solution[-1]

        # we negated the objective function, so negate it back
        objective_value: float = -h.getInfo().objective_function_value

        if debug:
            print(f"{i}: {w_star}, {objective_value:g}")

        # assess which constraints are currently active
        active_const: bool = False
        constraints = h.getBasis().row_status
        for c in range(len(constraints)-2):  # first two are sum-to-half
            if constraints[c+2] == highspy.HighsBasisStatus.kBasic:
                active_const = True
                if debug:
                    print(f"P{c} active")  # don't have slack values
        if debug and not active_const:
            print("No active constraints!")

        # z coefficient for the new constraint
        alpha: float = sqrt(w_star.transpose()@omega@w_star)

        # if gap between z and w'Omega w has converged, done
        if abs(z_star - alpha) < robust_gap_tol:
            break

        # add a new plane to the approximation of the uncertainty cone
        num_nz: int = dimension + 1  # HACK assuming entirely dense
        index: range = range(dimension + 1)
        value: npt.NDArray[np.float64] = np.append(-omega@w_star, alpha)
        h.addRow(0, inf, num_nz, index, value)

    # final value of solution is the z value, return separately
    return w_star, z_star, objective_value


# make highs_robust_genetics(...) an alias of the fastest method
highs_robust_genetics = highs_robust_genetics_sqp