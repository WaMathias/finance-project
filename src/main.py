import datetime
import time

import numpy as np

from src.analyzer.StockAnalyzer import StockAnalyzer
from src.data.TickerDataViewer import TickerDataViewer
from modules.models.capm import compute_capm, plot_capm
from src.data.DataLoader import *
from src.modules.models.cvar import analyze_portfolio_cvar
from src.modules.models.var import analyze_portfolio_var


def analyze_capm_for_tickers(tickers: list, market_index: list[str], start_date, end_date, debug=False):
    """
    Führt die CAPM-Analyse für eine Liste von Tickers durch.
    """
    from src.data.DataLoader import get_daily_returns, get_risk_free_rate

    market_returns_df = get_daily_returns(market_index, start_date, end_date)
    risk_free = get_risk_free_rate()

    if debug:
        print(f"[DEBUG] Risk-free rate (annual): {risk_free}")
        print(f"[DEBUG] Market returns type: {type(market_returns_df)}")
        print(
            f"[DEBUG] Market returns shape: {market_returns_df.shape if hasattr(market_returns_df, 'shape') else 'no shape'}")
        if hasattr(market_returns_df, 'columns'):
            print(f"[DEBUG] Market returns columns: {list(market_returns_df.columns)}")

    if isinstance(market_returns_df, pd.DataFrame):
        if market_index in market_returns_df.columns:
            market_returns = market_returns_df[market_index]
        else:
            market_returns = market_returns_df.iloc[:, 0]
            if debug:
                print(
                    f"[DEBUG] Market ticker '{market_index}' nicht in Spalten gefunden, nehme erste Spalte: {market_returns_df.columns[0]}")
    else:
        market_returns = market_returns_df

    if debug:
        print(f"[DEBUG] Market returns nach Konvertierung: type={type(market_returns)}, length={len(market_returns)}")

    for ticker in tickers:
        stock_returns_df = get_daily_returns(ticker, start_date, end_date)

        if debug:
            print(f"\n[DEBUG] Processing {ticker}")
            print(f"[DEBUG] Stock returns type: {type(stock_returns_df)}")
            print(
                f"[DEBUG] Stock returns shape: {stock_returns_df.shape if hasattr(stock_returns_df, 'shape') else 'no shape'}")
            if hasattr(stock_returns_df, 'columns'):
                print(f"[DEBUG] Stock returns columns: {list(stock_returns_df.columns)}")

        if isinstance(stock_returns_df, pd.DataFrame):
            if ticker in stock_returns_df.columns:
                stock_returns = stock_returns_df[ticker]
            else:
                stock_returns = stock_returns_df.iloc[:, 0]
                if debug:
                    print(
                        f"[DEBUG] Stock ticker '{ticker}' nicht in Spalten gefunden, nehme erste Spalte: {stock_returns_df.columns[0]}")
        else:
            stock_returns = stock_returns_df

        if debug:
            print(f"[DEBUG] Stock returns nach Konvertierung: type={type(stock_returns)}, length={len(stock_returns)}")

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

    analyze_capm_for_tickers(tickers, market_index='RHM.DE', start_date=start_date, end_date=end_date)

    weights = np.array([0.4, 0.3, 0.3])

    returns_df = get_daily_returns(tickers, start_date, end_date)
    var_results = analyze_portfolio_var(returns_df, weights)
    cvar_results = analyze_portfolio_cvar(returns_df, weights)



if __name__ == "__main__":
    main()
