import time
import datetime

from DataLoader import FinancialDataLoader
from RiskAnalyzer import PortfolioRiskAnalyzer
from TickerDataViewer import TickerDataViewer


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
    end_date = datetime.datetime.fromtimestamp(time.time())

    run_analysis(tickers, start_date, end_date, weights=None, alpha=0.95)

    viewer = TickerDataViewer(tickers)
    viewer.print_summaries()
    viewer.plot_chart(start_date, end_date)

