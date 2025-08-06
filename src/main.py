import datetime
import time

from src.analyzer.StockAnalyzer import StockAnalyzer
from src.data.TickerDataViewer import TickerDataViewer
from modules.models.capm import compute_capm, plot_capm
from src.data.DataLoader import *


def analyze_capm_for_tickers(tickers: list, market_index: str, start_date, end_date, debug=False):
    """
    Führt die CAPM-Analyse für eine Liste von Tickers durch.
    """
    from src.data.DataLoader import get_daily_returns, get_risk_free_rate

    market_returns = get_daily_returns(market_index, start_date, end_date)
    risk_free = get_risk_free_rate()

    if debug:
        print(f"[DEBUG] Risk-free rate (annual): {risk_free}")

    for ticker in tickers:
        stock_returns = get_daily_returns(ticker, start_date, end_date)
        capm_result = compute_capm(stock_returns, market_returns, risk_free, debug=debug)
        print(f"\n--- CAPM Analysis for {ticker} ---")
        for key, val in capm_result.items():
            print(f"{key}: {val:.4f}")
        plot_capm(stock_returns, market_returns, risk_free, ticker)


def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime.fromtimestamp(time.time())

    # Load price data
    loader = FinancialDataLoader(tickers, start_date, end_date)
    prices = loader.load_price_data()
    print(prices.tail())

    # Analyze each stock
    for ticker in tickers:
        analyzer = StockAnalyzer(ticker)
        analyzer.display_all()

    # Plot prices
    viewer = TickerDataViewer(tickers)
    viewer.plot_price_chart(start_date, end_date)

    analyze_capm_for_tickers(tickers, market_index='MSFT', start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
