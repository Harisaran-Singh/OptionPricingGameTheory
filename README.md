#  Option Pricing & Game Theory

This project combines **Binomial Tree Option Pricing** with **Game Theory (Nash Equilibrium)** to explore strategic financial decision-making. 

Instead of relying solely on traditional deterministic execution paths, this model calculates execution probabilities by framing the option holder vs. option writer dynamic as a zero-sum game.

##  Features
* **Live Data Integration:** Automatically fetches historical stock prices and option chains via `yfinance`.
* **Binomial Tree Pricing:** Custom implementation of a discrete-time binomial tree for dynamic asset pricing.
* **Game-Theoretic Execution:** Uses linear programming (`scipy.optimize.linprog`) to calculate the Nash Equilibrium strategy for holding vs. executing an option contract.
* **Historical Backtesting:** Tests the Nash strategy probabilities against real historical stock movements to track cumulative profit/loss.
* **Sensitivity Analysis:** Analyzes model robustness by observing how variations in volatility ($\sigma$), risk-free rates ($r$), and tree granularity affect average payoffs.

##  Project Structure
```text
.
├── core/
│   ├── data_loader.py       # Handles Yahoo Finance data fetching & volatility
│   ├── pricing_model.py     # Binomial tree and payoff matrix logic
│   └── game_theory.py       # Nash equilibrium linear programming solver
├── backtest/
│   └── strategy.py          # Backtesting and sensitivity analysis tools
├── requirements.txt         # Project dependencies
├── main.py                  # Entry point to run the pipeline
└── README.md
