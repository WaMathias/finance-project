import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


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


class TickerDataViewer:
    def __init__(self, tickers):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers

    def print_summaries(self):
        for ticker in self.tickers:
            print("=" * 60)
            print(f"Daten für: {ticker}")
            print("-" * 60)

            ticker_obj = yf.Ticker(ticker)

            historical_data = ticker_obj.history(period='max')
            print("→ Historische Daten (Kursverlauf):")
            print(historical_data.tail(5))

            financial_data = ticker_obj.financials
            print("\n→ Finanzdaten:")
            print(financial_data.iloc[:, :2] if not financial_data.empty else "Keine Finanzdaten verfügbar.")

            actions = ticker_obj.actions
            print("\n→ Kapitalmaßnahmen (Splits, Dividenden):")
            print(actions.tail(5) if not actions.empty else "Keine Kapitalmaßnahmen verfügbar.")

            print("=" * 60 + "\n")

    def plot_chart(self, start_date, end_date):
        data = yf.download(self.tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(data[ticker], label=f'Schlusskurs {ticker}')
            mean_price = data[ticker].mean()
            plt.axhline(mean_price, color='red', linestyle='--', label=f'Durchschnitt {ticker}: {mean_price:.2f} EUR')

        plt.title(f"Schlusskurse von {', '.join(self.tickers)} von {start_date.date()} bis {end_date.date()}")
        plt.xlabel("Datum")
        plt.ylabel("Preis in EUR")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


def run_analysis(tickers, start_date, end_date, weights=None, alpha=0.95):
    loader = FinancialDataLoader(tickers, start_date, end_date)
    prices = loader.load_price_data()

    analyzer = PortfolioRiskAnalyzer(prices, weights, alpha)
    analyzer.compute_var_cvar()
    analyzer.report()
    analyzer.plot_loss_distribution()



if __name__ == "__main__":
    tickers = ['AAPL', 'RHM.DE']
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime(2025, 8, 5)

    run_analysis(tickers, start_date, end_date, weights=None, alpha=0.95)

    viewer = TickerDataViewer(tickers)
    viewer.print_summaries()
    viewer.plot_chart(start_date, end_date)

