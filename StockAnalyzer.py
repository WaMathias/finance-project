# StockAnalyzer.py

import yfinance as yf

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.hist = self.stock.history(period="1y")
        self.returns = self.hist["Close"].pct_change().dropna()
        self.info = self.stock.info

    def analyze(self):
        print(f"\n📊 Analysis for: {self.ticker.upper()}")
        self.display_price_data()
        self.display_dividend_info()
        self.display_company_structure()
        self.display_valuation_metrics()
        self.display_fundamental_data()
        self.display_company_description()
        self.display_dividend_history()
        print("\n✅ Analysis complete.")

    def display_price_data(self):
        volatility = self.returns.std() * (252 ** 0.5)
        print("\nPrice Performance (Last Year):")
        print(f"Current Price: {self.hist['Close'][-1]:.2f}")
        print(f"52-Week High: {self.hist['Close'].max():.2f}")
        print(f"52-Week Low: {self.hist['Close'].min():.2f}")
        print(f"Annual Return: {self.returns.sum():.2%}")
        print(f"Annualized Volatility: {volatility:.2%}")

    def display_dividend_info(self):
        print("\nDividends:")
        print(f"Dividend Yield: {self.info.get('dividendYield', 'n/a')}")
        print(f"Dividend per Share: {self.info.get('dividendRate', 'n/a')}")
        print(f"Payout Ratio: {self.info.get('payoutRatio', 'n/a')}")

    def display_company_structure(self):
        print("\nCompany Structure:")
        print(f"Market Capitalization: {self.info.get('marketCap', 'n/a')}")
        print(f"Shares Outstanding: {self.info.get('sharesOutstanding', 'n/a')}")
        print(f"Beta: {self.info.get('beta', 'n/a')}")

    def display_valuation_metrics(self):
        print("\nValuation Metrics:")
        print(f"PE Ratio (Trailing): {self.info.get('trailingPE', 'n/a')}")
        print(f"PE Ratio (Forward): {self.info.get('forwardPE', 'n/a')}")
        print(f"Price/Sales: {self.info.get('priceToSalesTrailing12Months', 'n/a')}")
        print(f"Price/Book: {self.info.get('priceToBook', 'n/a')}")
        print(f"Enterprise Value/Revenue: {self.info.get('enterpriseToRevenue', 'n/a')}")
        print(f"Enterprise Value/EBITDA: {self.info.get('enterpriseToEbitda', 'n/a')}")

    def display_fundamental_data(self):
        print("\nFundamentals:")
        print(f"Revenue (ttm): {self.info.get('totalRevenue', 'n/a')}")
        print(f"Gross Profit: {self.info.get('grossProfits', 'n/a')}")
        print(f"Net Income: {self.info.get('netIncomeToCommon', 'n/a')}")
        print(f"EBITDA: {self.info.get('ebitda', 'n/a')}")
        print(f"Operating Income: {self.info.get('operatingIncome', 'n/a')}")
        print(f"Free Cash Flow: {self.info.get('freeCashflow', 'n/a')}")
        print(f"Total Debt: {self.info.get('totalDebt', 'n/a')}")
        print(f"Current Ratio: {self.info.get('currentRatio', 'n/a')}")
        print(f"Quick Ratio: {self.info.get('quickRatio', 'n/a')}")
        print(f"Return on Equity (ROE): {self.info.get('returnOnEquity', 'n/a')}")
        print(f"Return on Assets (ROA): {self.info.get('returnOnAssets', 'n/a')}")
        print(f"Return on Capital (ROC): {self.info.get('returnOnCapitalEmployed', 'n/a')}")
        print(f"Gross Margin: {self.info.get('grossMargins', 'n/a')}")
        print(f"Operating Margin: {self.info.get('operatingMargins', 'n/a')}")
        print(f"Profit Margin: {self.info.get('profitMargins', 'n/a')}")

    def display_company_description(self):
        print("\nCompany Description:")
        print(self.info.get("longBusinessSummary", "No description available."))

    def display_dividend_history(self):
        dividends = self.stock.dividends
        if not dividends.empty:
            print("\n📜 Dividend History:")
            print(dividends.tail(5))
