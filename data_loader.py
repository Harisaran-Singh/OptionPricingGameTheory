import yfinance as yf
import numpy as np

def fetch_data(ticker, start_date, end_date):
    """Fetch stock and option data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    stock_prices = stock.history(start=start_date, end=end_date)['Close']

    # Get options data (nearest expiration)
    options_dates = stock.options
    if not options_dates:
        raise ValueError(f"No options data available for the ticker: {ticker}")

    option_chain = stock.option_chain(options_dates[0])
    calls = option_chain.calls
    puts = option_chain.puts

    return stock_prices, calls, puts

def calculate_volatility(prices):
    """Calculate annualized historical volatility from log returns."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)