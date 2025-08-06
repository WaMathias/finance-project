import time

import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

class PortfolioRiskAnalyzer:
    def __init__(self, prices: pd.DataFrame, weights: np.ndarray = None, alpha: float = 0.95):
        self.prices = prices
        self.alpha = alpha
        self.start_date = prices.index.min()  # Startdatum aus den Preisen
        self.end_date = prices.index.max()    # Enddatum aus den Preisen

        self.rets = self._calculate_returns()
        self.weights = self._prepare_weights(weights)
        self._validate_dimensions()

        self.portfolio_returns = self._calculate_portfolio_returns()
        self.losses = -self.portfolio_returns

        self.var = None
        self.cvar = None

    def _calculate_returns(self) -> pd.DataFrame:
        return self.prices.pct_change(fill_method=None).dropna()


    def _prepare_weights(self, weights: np.ndarray) -> np.ndarray:
        num_assets = self.rets.shape[1]
        if weights is None:
            weights = np.ones(num_assets) / num_assets  # Gleichverteilung
        else:
            weights = np.array(weights)
            if weights.sum() != 1.0:
                weights = weights / weights.sum()  # Normalisierung
        return weights

    def _validate_dimensions(self):
        if len(self.weights) != self.rets.shape[1]:
            raise ValueError(f"Fehler: Anzahl der Gewichte ({len(self.weights)}) passt nicht zur Anzahl der Assets ({self.rets.shape[1]}).")

    def _calculate_portfolio_returns(self) -> pd.Series:
        return self.rets.dot(self.weights)

    def compute_var_cvar(self):
        var_threshold = np.percentile(self.losses, (1 - self.alpha) * 100)
        cvar = self.losses[self.losses >= var_threshold].mean()
        self.var = var_threshold
        self.cvar = cvar

    def plot_loss_distribution(self):
        if self.var is None or self.cvar is None:
            raise RuntimeError("Zuerst compute_var_cvar() aufrufen.")
        plt.figure(figsize=(10, 6))
        plt.hist(self.losses, bins=100, density=True, alpha=0.6, color='skyblue')
        plt.axvline(self.var, color='red', linestyle='--', label=f'VaR {self.alpha:.0%} = {self.var:.4%}')
        plt.axvline(self.cvar, color='black', linestyle=':', label=f'CVaR = {self.cvar:.4%}')
        plt.title("Verlustverteilung")
        plt.xlabel("Tagesverlust")
        plt.ylabel("Dichte")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def report(self):
        asset_info = f"{len(self.weights)} Asset{'s' if len(self.weights) > 1 else ''}"
        print(f"Portfolio mit {asset_info} und α = {self.alpha:.0%}")
        print(f'→ VaR:  {self.var:.4%}')
        print(f'→ CVaR: {self.cvar:.4%}')

    def calculate_capm(stock_returns, market_returns, risk_free_rate):
        excess_stock = stock_returns - risk_free_rate
        excess_market = market_returns - risk_free_rate

        X = sm.add_constant(excess_market)
        model = sm.OLS(excess_stock, X).fit()

        beta = model.params[1]
        alpha = model.params[0]
        return alpha, beta, model
