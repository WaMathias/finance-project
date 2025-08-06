import datetime
import time

from src.data.DataLoader import FinancialDataLoader
from src.analyzer.StockAnalyzer import StockAnalyzer
from src.data.TickerDataViewer import TickerDataViewer

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

if __name__ == "__main__":
    main()
