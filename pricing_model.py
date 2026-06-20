import numpy as np

def binomial_tree(S0, K, T, steps, r, sigma):
    """Constructs the binomial tree for stock prices."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    price_tree = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = S0 * (u**(i - j)) * (d**j)

    return price_tree, p, u, d

def payoff_matrix(price_tree, K, is_call=False):
    """Calculates the payoff matrix for the option."""
    steps = price_tree.shape[1] - 1
    payoff_tree = np.copy(price_tree)
    
    for i in range(steps + 1):
        for j in range(i + 1):
            if is_call:
                payoff_tree[j, i] = max(payoff_tree[j, i] - K, 0)
            else:
                payoff_tree[j, i] = max(K - payoff_tree[j, i], 0)
                
    return np.maximum(payoff_tree, 0)