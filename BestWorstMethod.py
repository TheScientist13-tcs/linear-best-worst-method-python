import pyomo.environ as pyo
import numpy as np

def _build_model(n, bo_vec, ow_vec, best_idx, worst_idx):

    best_idx = best_idx + 1
    worst_idx = worst_idx + 1
    bo_vec = {j+1: bo_vec[j] for j in range(n)}
    ow_vec = {j+1: ow_vec[j] for j in range(n)}

    model = pyo.ConcreteModel()

    # Index set of criteria
    model.C = pyo.RangeSet(1, n)

    # Variables: weights of criteria (non-negative)
    model.w = pyo.Var(model.C, domain=pyo.NonNegativeReals)

    # Variable xi (max deviation)
    model.xi = pyo.Var(domain=pyo.NonNegativeReals)

    # Constraint: sum of weights equals 1
    model.sum_weights = pyo.Constraint(expr=sum(model.w[j] for j in model.C) == 1)

    model.best_constraints = pyo.ConstraintList()
    model.worst_constraints = pyo.ConstraintList()

    # Add constraints: |w_B - a_Bj * w_j| <= xi
    for j in model.C:
        model.best_constraints.add(model.w[best_idx] - bo_vec[j] * model.w[j] <= model.xi)
        model.best_constraints.add(bo_vec[j] * model.w[j] - model.w[best_idx] <= model.xi)

    # Add constraints: |w_j - a_jW * w_W| <= xi
    for j in model.C:
        model.worst_constraints.add(model.w[j] - ow_vec[j] * model.w[worst_idx] <= model.xi)
        model.worst_constraints.add(ow_vec[j] * model.w[worst_idx] - model.w[j] <= model.xi)

    # Objective: minimize xi
    model.obj = pyo.Objective(expr=model.xi, sense=pyo.minimize)

    return model

def solve_bwm(n, bo_vec, ow_vec, best_idx, worst_idx):
    """
    n: number of criteria
    bo_vec: best-to-others vector 
    ow_vec: others-to-worst vector 
    best_idx: index of best criterion (0-based)
    worst: index of worst criterion (0-based)
    """
    model = _build_model(n, bo_vec, ow_vec, best_idx, worst_idx)
    solver = pyo.SolverFactory('glpk')
    result = solver.solve(model)
    print("Status:", result.solver.status)
    print("Termination Condition:", result.solver.termination_condition)
    weights = np.zeros(n)

    for j in model.C:
        weights[j-1] = pyo.value(model.w[j])

    return {"weights":np.array(weights), "xi":pyo.value(model.xi)}

def consistency_ratio(best_to_worst_value, xi_opt):
    consistency_index_values = [0, 0.44, 1, 1.63, 2.3, 3, 3.73, 4.47, 5.23]
    consistency_index = consistency_index_values[best_to_worst_value - 1]
    return xi_opt/consistency_index
    