# cvar.py

import numpy as np
import pandas as pd
from scipy.stats import norm, t

import logging

from src.modules.models.var import calculate_historical_var, calculate_parametric_var, calculate_monte_carlo_var

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_cvar(returns: pd.Series, var: float) -> float:
    """Berechnet den Conditional Value at Risk (CVaR) gegebenen VaR."""
    cvar = -returns[returns < -var].mean()
    logger.debug(f"CVaR (gegeben VaR={var:.4f}): {cvar:.4f}")
    return cvar


def compute_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Berechnet gewichtete Portfolio-Renditen."""
    if not np.isclose(weights.sum(), 1):
        raise ValueError("Die Summe der Gewichte muss 1 ergeben.")
    return returns_df.dot(weights)


def analyze_var_cvar(returns: pd.Series, name: str = "Asset", confidence_level: float = 0.95, show_plot: bool = True):
    """Analysiert alle VaR/CVaR-Methoden für eine einzelne Renditeserie."""
    results = {}

    var_hist = calculate_historical_var(returns, confidence_level)
    var_param = calculate_parametric_var(returns, confidence_level)
    var_mc = calculate_monte_carlo_var(returns, confidence_level)

    cvar_hist = calculate_cvar(returns, var_hist)
    cvar_param = calculate_cvar(returns, var_param)
    cvar_mc = calculate_cvar(returns, var_mc)

    results["CVaR (Historisch)"] = cvar_hist
    results["CVaR (Parametrisch)"] = cvar_param
    results["CVaR (Monte Carlo)"] = cvar_mc

    return results


def analyze_portfolio_cvar(returns_df: pd.DataFrame, weights: np.ndarray, name: str = "Portfolio",
                                confidence_level: float = 0.95, show_plot: bool = True):
    """Analysiert VaR/CVaR eines Portfolios mit gewichteten Renditen."""
    portfolio_returns = compute_portfolio_returns(returns_df, weights)
    return analyze_var_cvar(portfolio_returns, name=name, confidence_level=confidence_level, show_plot=show_plot)
