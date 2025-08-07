# var.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Berechnet den historischen Value at Risk (VaR)."""
    var = -np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    logger.debug(f"Historischer VaR @ {confidence_level:.0%}: {var:.4f}")
    return var


def calculate_parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Berechnet den parametrischen Value at Risk (Normalverteilung)."""
    mu = returns.mean()
    sigma = returns.std()
    z_score = abs(np.percentile(np.random.normal(0, 1, 10**6), (1 - confidence_level) * 100))
    var = -(mu - z_score * sigma)
    logger.debug(f"Parametrischer VaR @ {confidence_level:.0%}: {var:.4f}")
    return var


def calculate_monte_carlo_var(returns: pd.Series, confidence_level: float = 0.95, simulations: int = 10000) -> float:
    """Berechnet den Monte-Carlo Value at Risk (VaR)."""
    mu = returns.mean()
    sigma = returns.std()
    simulated_returns = np.random.normal(mu, sigma, simulations)
    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    logger.debug(f"Monte Carlo VaR @ {confidence_level:.0%}: {var:.4f}")
    return var


def calculate_var_historic(returns: pd.Series, confidence_level: float = 0.95) -> float:
    var = -np.percentile(returns, (1 - confidence_level) * 100)
    logger.debug(f"Historischer VaR @ {confidence_level:.0%}: {var:.4f}")
    return var


def calculate_var_monte_carlo_heavy_tail(returns: pd.Series, confidence_level: float = 0.95,
                                         simulations: int = 10000, df: int = 5) -> float:
    mu = returns.mean()
    sigma = returns.std()
    simulated_returns = mu + sigma * t.rvs(df, size=simulations)
    simulated_returns = inject_crash_events(simulated_returns)
    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    logger.debug(f"Monte Carlo VaR (Heavy Tail, df={df}) @ {confidence_level:.0%}: {var:.4f}")
    return var


def inject_crash_events(simulated_returns: np.ndarray, crash_prob: float = 0.01, crash_magnitude: float = -0.2) -> np.ndarray:
    n = len(simulated_returns)
    crashes = np.random.choice([0, 1], size=n, p=[1 - crash_prob, crash_prob])
    simulated_returns += crashes * crash_magnitude
    logger.debug(f"Crashs injiziert: {crashes.sum()} bei Crash-Magnitude={crash_magnitude}")
    return simulated_returns

def scale_to_annual(value: float, periods_per_year: int = 252) -> float:
    return value * np.sqrt(periods_per_year)


def plot_var_histogram(returns: pd.Series, var_dict: dict, title: str):
    """Visualisiert die Renditeverteilung mit VaR-Linien."""
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.6, color='grey', edgecolor='black')
    for label, value in var_dict.items():
        plt.axvline(-value, linestyle='--', label=f"{label}: {value:.4f}")
    plt.title(f"Renditeverteilung & VaR – {title}")
    plt.xlabel("Rendite")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_var(returns: pd.Series,
                name: str = "Asset",
                confidence_level: float = 0.95,
                show_plot: bool = True,
                annualize: bool = False,
                include_heavy_tail: bool = True) -> dict:
    """
    Analysiert Value at Risk (VaR) eines Assets mit verschiedenen Methoden.
    Nutzt ausschließlich die Funktionen, die bereitgestellt wurden.
    """

    results = {}

    # Klassische Methoden
    var_hist = calculate_historical_var(returns, confidence_level)
    var_param = calculate_parametric_var(returns, confidence_level)
    var_mc = calculate_monte_carlo_var(returns, confidence_level)

    # Optional: Monte Carlo mit Fat Tails und Crashes
    if include_heavy_tail:
        var_mc_heavy = calculate_var_monte_carlo_heavy_tail(returns, confidence_level)
        results["VaR (Monte Carlo Heavy Tail)"] = scale_to_annual(var_mc_heavy) if annualize else var_mc_heavy

    # Ergebnisse eintragen
    results["VaR (Historisch)"] = scale_to_annual(var_hist) if annualize else var_hist
    results["VaR (Parametrisch)"] = scale_to_annual(var_param) if annualize else var_param
    results["VaR (Monte Carlo)"] = scale_to_annual(var_mc) if annualize else var_mc

    # Visualisierung
    if show_plot:
        var_lines = {
            "Historisch": results["VaR (Historisch)"],
            "Parametrisch": results["VaR (Parametrisch)"],
            "Monte Carlo": results["VaR (Monte Carlo)"],
        }
        if include_heavy_tail:
            var_lines["Monte Carlo Heavy Tail"] = results["VaR (Monte Carlo Heavy Tail)"]

        plot_var_histogram(returns, var_lines, title=name)

    # Logging
    logger.info(f"Analyseergebnisse für {name}:")
    for method, value in results.items():
        logger.info(f"{method}: {value:.4f}")

    return results



def compute_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Berechnet gewichtete Portfolio-Renditen."""
    if not np.isclose(weights.sum(), 1):
        raise ValueError("Die Summe der Gewichte muss 1 ergeben.")
    return returns_df.dot(weights)


def analyze_portfolio_var(returns_df: pd.DataFrame, weights: np.ndarray, name: str = "Portfolio",
                                confidence_level: float = 0.95, show_plot: bool = True):
    """Analysiert VaR/CVaR eines Portfolios mit gewichteten Renditen."""
    portfolio_returns = compute_portfolio_returns(returns_df, weights)
    return analyze_var(portfolio_returns, name=name, confidence_level=confidence_level, show_plot=show_plot)
