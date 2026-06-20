import pandas as pd
from core.data_loader import fetch_data, calculate_volatility
from core.pricing_model import binomial_tree, payoff_matrix
from core.game_theory import calculate_nash_equilibrium
from backtest.strategy import backtest_strategy, plot_cumulative_profit, sensitivity_analysis

# 1. Initialization of Parameters
TICKER = "AAPL"
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"
STEPS = 100
R = 0.06
T = 1
IS_CALL = False  # Set to False to model a Put Option (as done in the original notebook)

def main():
    print(f"Fetching data for {TICKER}...")
    prices, calls, puts = fetch_data(TICKER, START_DATE, END_DATE)
    
    sigma = calculate_volatility(prices)
    S0 = prices.iloc[0]
    K = S0
    
    print(f"Calculated Annualized Volatility: {sigma:.4f}")
    
    # 2. Build Binomial Tree & Payoff Matrix
    print("Building Binomial Tree and calculating Payoff Matrix...")
    tree, p, u, d = binomial_tree(S0, K, T, STEPS, R, sigma)
    payoff_mat = payoff_matrix(tree, K, is_call=IS_CALL)
    
    # 3. Game Theory: Calculate Nash Equilibrium
    print("Solving zero-sum game for Nash Equilibrium...")
    strategy = calculate_nash_equilibrium(payoff_mat)
    print(f"Optimal Strategy Probabilities -> Hold: {strategy[0]:.4f}, Execute: {strategy[1]:.4f}")
    
    # 4. Backtesting
    print("Running Backtest on historical data...")
    backtest_results = backtest_strategy(prices, K, strategy, is_call=IS_CALL)
    total_profit = backtest_results['Profit'].sum()
    print(f"Total Backtest Profit: ${total_profit:.2f}")
    
    # Plot results
    plot_cumulative_profit(backtest_results)
    
    # 5. Sensitivity Analysis
    print("Running Sensitivity Analysis...")
    base_params = {
        "S0": S0, "K": K, "T": T, 
        "steps": STEPS, "r": R, "sigma": sigma
    }
    variations = {
        "sigma": [0.2, sigma, 0.3],     # Test volatility
        "steps": [50, 100, 200],        # Test tree granularity
        "r": [0.04, 0.06, 0.08]         # Test interest rates
    }
    
    sensitivity_results = sensitivity_analysis(base_params, variations, is_call=IS_CALL)
    print("\nSensitivity Analysis Results:")
    print(sensitivity_results.to_string())

if __name__ == "__main__":
    main()