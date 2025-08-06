import os
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE")
FMP_KEY = os.getenv("FMP")

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = self.stock.info
        self.hist = self.stock.history(period="1y")
        self.alpha_vantage_key = ALPHA_VANTAGE_KEY
        self.fmp_key = FMP_KEY

    def get_yfinance_data(self):
        returns = self.hist['Close'].pct_change().dropna()
        data = {
            "current_price": self.hist['Close'][-1],
            "52w_high": self.hist['Close'].max(),
            "52w_low": self.hist['Close'].min(),
            "annual_return": returns.sum(),
            "dividend_yield": self.info.get('dividendYield'),
            "pe_ratio": self.info.get('trailingPE'),
            "market_cap": self.info.get('marketCap'),
            # weitere yfinance-Felder, die du brauchst ...
        }
        return data

    def get_alpha_vantage_income_statement(self):
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key missing!")
            return None

        url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={self.ticker}&apikey={self.alpha_vantage_key}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Alpha Vantage request failed")
            return None
        return response.json().get("annualReports", [])

    def get_fmp_ratios(self):
        if not self.fmp_key:
            print("FMP API key missing!")
            return None

        url = f"https://financialmodelingprep.com/api/v3/ratios/{self.ticker}?apikey={self.fmp_key}"
        response = requests.get(url)
        if response.status_code != 200:
            print("FMP request failed")
            return None
        return response.json()

    def display_basic_info(self):
        data = self.get_yfinance_data()
        print(f"\nBasic info for {self.ticker}:")
        for key, value in data.items():
            print(f"  {key}: {value}")

    def display_alpha_vantage_income(self):
        reports = self.get_alpha_vantage_income_statement()
        if not reports:
            print("No Alpha Vantage income statement data available.")
            return
        latest = reports[0]
        print(f"\nAlpha Vantage Income Statement (latest) for {self.ticker}:")
        for key, value in latest.items():
            print(f"  {key}: {value}")

    def display_fmp_ratios(self):
        ratios = self.get_fmp_ratios()
        if not ratios:
            print("No FMP ratios available.")
            return
        latest = ratios[0]
        print(f"\nFMP Ratios (latest) for {self.ticker}:")
        for key, value in latest.items():
            print(f"  {key}: {value}")

    def display_all(self):
        self.display_basic_info()
        self.display_alpha_vantage_income()
        self.display_fmp_ratios()
