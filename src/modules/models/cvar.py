import numpy as np
import pandas as pd
import logging

from src.modules.models.var import (
    calculate_historical_var,
    calculate_parametric_var,
    calculate_monte_carlo_var
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_cvar_from_var(returns: pd.Series, var: float) -> float:
    """Calculate CVaR given VaR."""
    cvar = -returns[returns < -var].mean()
    logger.debug(f"CVaR (given VaR={var:.4f}): {cvar:.4f}")
    return cvar

def calculate_historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    var = calculate_historical_var(returns, confidence_level)
    return calculate_cvar_from_var(returns, var)

def calculate_parametric_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    var = calculate_parametric_var(returns, confidence_level)
    return calculate_cvar_from_var(returns, var)

def calculate_monte_carlo_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    var = calculate_monte_carlo_var(returns, confidence_level)
    return calculate_cvar_from_var(returns, var)

def analyze_portfolio_cvar(returns_df: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95) -> dict:
    """Calculates CVaR for weighted portfolio returns."""
    portfolio_returns = returns_df.dot(weights)
    results = {
        "CVaR (Historisch)": calculate_historical_cvar(portfolio_returns, confidence_level),
        "CVaR (Parametrisch)": calculate_parametric_cvar(portfolio_returns, confidence_level),
        "CVaR (Monte Carlo)": calculate_monte_carlo_cvar(portfolio_returns, confidence_level)
    }
    return results
