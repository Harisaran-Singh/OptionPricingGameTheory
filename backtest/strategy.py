import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.pricing_model import binomial_tree, payoff_matrix
from core.game_theory import calculate_nash_equilibrium

def backtest_strategy(stock_prices, K, nash_strategy, is_call=False):
    """Simulates trading over historical data using the calculated Nash strategy probabilities."""
    results = []
    for date, price in stock_prices.items():
        # Action is chosen probabilistically based on the Nash Strategy
        action = np.random.choice(["Hold", "Execute"], p=nash_strategy)

        if action == "Execute":
            profit = np.maximum(price - K, 0) if is_call else np.maximum(K - price, 0)
        else:
            profit = 0
            
        results.append({
            "Date": date,
            "Stock_Price": price,
            "Action": action,
            "Profit": profit
        })
    return pd.DataFrame(results)

def plot_cumulative_profit(df):
    """Plots the cumulative profit generated during the backtest."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Cumulative_Profit'] = df['Profit'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Cumulative_Profit'], linestyle='-', color='green')
    plt.title('Cumulative Profit Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Profit', fontsize=12)
    plt.grid(True)
    plt.show()

def sensitivity_analysis(base_params, variations, is_call=False):
    """Analyzes the model's robustness by adjusting base parameters."""
    results = []
    for param, values in variations.items():
        for value in values:
            params = base_params.copy()
            params[param] = value

            # Rebuild model with varied parameters
            tree, p, u, d = binomial_tree(
                params['S0'], params['K'], params['T'], 
                params['steps'], params['r'], params['sigma']
            )
            payoff = payoff_matrix(tree, params['K'], is_call=is_call)

            try:
                nash_strategy = calculate_nash_equilibrium(payoff)
                results.append({
                    "Parameter": param,
                    "Value": value,
                    "Hold_Prob": nash_strategy[0],
                    "Execute_Prob": nash_strategy[1],
                    "Avg_Payoff": np.mean(payoff)
                })
            except ValueError:
                continue

    return pd.DataFrame(results)
