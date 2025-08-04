import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd


class FinancialDataLoader:
    def __init__(self, tickers, start_date, end_date):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def load_price_data(self) -> pd.DataFrame:
        print(f"Lade Daten für: {', '.join(self.tickers)}")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        close = data['Close']
        if isinstance(close, pd.Series):
            close = close.to_frame()
        close.columns = [str(t) for t in close.columns]
        return close


class PortfolioRiskAnalyzer:
    def __init__(self, prices: pd.DataFrame, weights: np.ndarray = None, alpha: float = 0.95):
        self.prices = prices
        self.alpha = alpha

        self.rets = self._calculate_returns()

        self.weights = self._prepare_weights(weights)
        self._validate_dimensions()

        self.portfolio_returns = self._calculate_portfolio_returns()
        self.losses = -self.portfolio_returns

        self.var = None
        self.cvar = None

    def _calculate_returns(self) -> pd.DataFrame:
        return self.prices.pct_change().dropna()

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


class TickerDataViewer:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.ticker_obj = yf.Ticker(ticker)

    def print_summary(self):
        print("=" * 60)
        print(f"📈 Daten für: {self.ticker}")
        print("-" * 60)

        historical_data = self.ticker_obj.history(period='max')
        print("→ Historische Daten (Kursverlauf):")
        print(historical_data.tail(5))

        financial_data = self.ticker_obj.financials
        print("\n→ Finanzdaten:")
        print(financial_data.iloc[:, :2])  # letze zwei Persioden

        actions = self.ticker_obj.actions
        print("\n→ Kapitalmaßnahmen (Splits, Dividenden):")
        print(actions.tail(5))

        print("=" * 60 + "\n")



def run_analysis(tickers, start_date, end_date, weights=None, alpha=0.95):
    loader = FinancialDataLoader(tickers, start_date, end_date)
    prices = loader.load_price_data()

    analyzer = PortfolioRiskAnalyzer(prices, weights, alpha)
    analyzer.compute_var_cvar()
    analyzer.report()
    analyzer.plot_loss_distribution()



if __name__ == "__main__":
    tickers = ['AAPL', 'RHM.DE']
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2025, 8, 1)
    run_analysis(tickers, start_date, end_date, weights=None, alpha=0.95)
    viewer = TickerDataViewer(tickers[0])
    viewer.print_summary()
