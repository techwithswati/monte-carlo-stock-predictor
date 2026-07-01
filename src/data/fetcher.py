"""
Market Data Fetcher
===================
Retrieves OHLCV data and computes return / volatility statistics
for use by the Monte Carlo engine.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed — using synthetic data fallback")


@dataclass
class MarketData:
    ticker: str
    current_price: float
    expected_return: float   # annualised
    volatility: float        # annualised
    history: Optional[pd.DataFrame] = None
    source: str = "yfinance"


def fetch_market_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    risk_free_rate: float = 0.0525,
) -> MarketData:
    """
    Download historical prices and derive μ and σ.

    Parameters
    ----------
    ticker        : Yahoo Finance ticker symbol
    period        : lookback window passed to yfinance (e.g. '1y', '2y', '5y')
    interval      : bar size  ('1d', '1wk', '1mo')
    risk_free_rate: used to compute excess-return estimate
    """
    if _YF_AVAILABLE:
        return _fetch_yfinance(ticker, period, interval, risk_free_rate)
    return _synthetic_data(ticker)


def _fetch_yfinance(ticker: str, period: str, interval: str, rf: float) -> MarketData:
    logger.info("Fetching %s | period=%s interval=%s", ticker, period, interval)
    tk = yf.Ticker(ticker)
    hist = tk.history(period=period, interval=interval, auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    prices = hist["Close"].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    trading_days = 252
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    annualised_return = float(mu_daily * trading_days + 0.5 * (sigma_daily ** 2) * trading_days)
    annualised_vol = float(sigma_daily * np.sqrt(trading_days))
    current_price = float(prices.iloc[-1])

    logger.info(
        "Market params | %s price=%.2f μ=%.4f σ=%.4f",
        ticker, current_price, annualised_return, annualised_vol,
    )
    return MarketData(
        ticker=ticker,
        current_price=current_price,
        expected_return=annualised_return,
        volatility=annualised_vol,
        history=hist,
        source="yfinance",
    )


def _synthetic_data(ticker: str) -> MarketData:
    """Fallback with realistic PE-grade default parameters."""
    defaults = {
        "AAPL": (185.0, 0.18, 0.28),
        "MSFT": (420.0, 0.22, 0.25),
        "NVDA": (900.0, 0.45, 0.55),
        "SPY":  (510.0, 0.10, 0.17),
        "TSLA": (250.0, 0.30, 0.65),
    }
    price, mu, sigma = defaults.get(ticker.upper(), (100.0, 0.12, 0.25))
    logger.warning("Using synthetic params for %s", ticker)
    return MarketData(
        ticker=ticker,
        current_price=price,
        expected_return=mu,
        volatility=sigma,
        source="synthetic",
    )
