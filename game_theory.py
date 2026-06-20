import numpy as np
from scipy.optimize import linprog

def calculate_nash_equilibrium(payoff_matrix):
    """Calculates probabilities of holding vs executing using a zero-sum game formulation."""
    n = payoff_matrix.shape[0]

    # Variables for Linear programming
    c = np.ones(n)  # Minimize sum(x)
    A_ub = -payoff_matrix.T  # Constraints: payoff.T * x <= 0
    b_ub = np.zeros(n)
    A_eq = np.ones((1, n))  # Condition for probability
    b_eq = np.ones(1)
    bounds = [(0, None)] * n

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if not res.success:
        raise ValueError("Failed to solve linear programming for Nash Equilibrium.")

    strategy = res.x
    strategy = np.maximum(strategy, 0)  # Ensure non-negative
    strategy /= strategy.sum()  # Normalize

    positive_payoffs = payoff_matrix[payoff_matrix > 0]
    if len(positive_payoffs) > 0:
        execute_prob = np.mean(positive_payoffs) / np.max(positive_payoffs)
    else:
        execute_prob = 0

    # Ensure probabilities are valid
    execute_prob = np.clip(execute_prob, 0, 1)
    hold_prob = 1 - execute_prob

    return np.array([hold_prob, execute_prob])