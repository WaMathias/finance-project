import time
import datetime

import RiskAnalyzer
from DataLoader import FinancialDataLoader
from RiskAnalyzer import PortfolioRiskAnalyzer
from StockAnalyzer import StockAnalyzer
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
    RiskAnalyzer.PortfolioRiskAnalyzer.calculate_capm(stock_returns=, market_returns=, risk_free_rate=)

    StockAnalyzer.analyze()
    StockAnalyzer.display_price_data()
    StockAnalyzer.display_dividend_info()
    StockAnalyzer.display_company_structure()
    StockAnalyzer.display_valuation_metrics()
    StockAnalyzer.display_fundamental_data()
    StockAnalyzer.display_company_description()
    StockAnalyzer.display_dividend_history()

    viewer = TickerDataViewer(tickers)
    viewer.plot_chart(start_date, end_date)
