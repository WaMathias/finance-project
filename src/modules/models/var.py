import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Historical VaR based on empirical quantiles."""
    var = -np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    logger.debug(f"Historical VaR @ {confidence_level:.0%}: {var:.4f}")
    return var

def calculate_parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Parametric VaR using normal distribution."""
    mu = returns.mean()
    sigma = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var = -(mu + z_score * sigma)
    logger.debug(f"Parametric VaR @ {confidence_level:.0%}: {var:.4f}")
    return var

def calculate_monte_carlo_var(returns: pd.Series, confidence_level: float = 0.95, simulations: int = 100_000) -> float:
    """Monte Carlo VaR using normal distribution sampling."""
    mu = returns.mean()
    sigma = returns.std()
    simulated_returns = np.random.normal(mu, sigma, simulations)
    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    logger.debug(f"Monte Carlo VaR @ {confidence_level:.0%}: {var:.4f}")
    return var

def analyze_portfolio_var(returns_df: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95) -> dict:
    """Calculates VaR for weighted portfolio returns."""
    portfolio_returns = returns_df.dot(weights)
    results = {
        "VaR (Historisch)": calculate_historical_var(portfolio_returns, confidence_level),
        "VaR (Parametrisch)": calculate_parametric_var(portfolio_returns, confidence_level),
        "VaR (Monte Carlo)": calculate_monte_carlo_var(portfolio_returns, confidence_level)
    }
    return results
