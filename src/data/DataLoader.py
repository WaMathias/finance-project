import datetime
import os
from dotenv import load_dotenv
import requests
import yfinance as yf
import pandas as pd
from pandas import DataFrame

# Load API keys from .env
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE")
FMP_API_KEY = os.getenv("FMP")

class FinancialDataLoader:
    def __init__(self, tickers, start_date, end_date):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def load_price_data(self) -> pd.DataFrame:
        print(f"Loading price data for: {', '.join(self.tickers)}")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        close = data['Close']
        if isinstance(close, pd.Series):
            close = close.to_frame()
        close.columns = [str(t) for t in close.columns]
        return close


# ======================== PRICE AND RETURN DATA ============================

def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical adjusted close prices from yfinance."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return data["Close"]


def get_daily_returns(ticker: list[str], start: datetime.datetime, end: datetime.datetime) -> DataFrame:
    """Calculate daily returns from price data."""
    prices = get_price_data(ticker, start, end)
    return prices.pct_change().dropna()


# ======================== FUNDAMENTAL DATA ============================

def get_fundamentals_yf(ticker: str) -> dict:
    """Retrieve fundamental data from yfinance."""
    stock = yf.Ticker(ticker)
    return {
        "financials": stock.financials,
        "balance_sheet": stock.balance_sheet,
        "cashflow": stock.cashflow,
        "info": stock.info
    }


def get_fundamentals_fmp(ticker: str) -> dict:
    """Fetch financial statements from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={FMP_API_KEY}&limit=5"
    income = requests.get(url).json()

    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={FMP_API_KEY}&limit=5"
    balance = requests.get(url).json()

    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={FMP_API_KEY}&limit=5"
    cashflow = requests.get(url).json()

    return {
        "income_statements": income,
        "balance_sheets": balance,
        "cash_flows": cashflow
    }


# ======================== VALUATION METRICS ============================

def get_ratios_fmp(ticker: str) -> dict:
    """Retrieve key valuation ratios from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={FMP_API_KEY}&limit=5"
    return requests.get(url).json()


def get_ratios_yf(ticker: str) -> dict:
    """Retrieve valuation ratios from yfinance."""
    stock = yf.Ticker(ticker)
    return stock.info


# ======================== DIVIDENDS & STOCK ACTIONS ============================

def get_dividends(ticker: str) -> pd.Series:
    """Get dividend history from yfinance."""
    return yf.Ticker(ticker).dividends


def get_stock_actions(ticker: str) -> pd.DataFrame:
    """Get stock splits and dividends from yfinance."""
    return yf.Ticker(ticker).actions


# ======================== MARKET & RISK-FREE RATE ============================

def get_market_data(ticker: str, start: str, end: str) -> pd.Series:
    """Get market index returns (e.g., ^GSPC for S&P 500)."""
    return get_daily_returns(ticker, start, end)


def get_risk_free_rate(source: str = "alpha_vantage") -> float:
    if source == "alpha_vantage":
        url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"AlphaVantage API returned status {response.status_code}, using fallback risk-free rate.")
            return 0.015  # z.B. 1,5% als Fallbackwert

        data = response.json()
        print("DEBUG AlphaVantage response:", data)
        try:
            latest = data['data'][0]
            return float(latest['value']) / 100
        except (KeyError, IndexError):
            print("Fehler beim Parsen der AlphaVantage API Antwort, benutze Fallbackwert.")
            return 0.015

    elif source == "fmp":
        url = f"https://financialmodelingprep.com/api/v4/treasury?apikey={FMP_API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"FMP API returned status {response.status_code}, using fallback risk-free rate.")
            return 0.015

        data = response.json()
        print("DEBUG FMP response:", data)
        try:
            return float(data[0]['year10']) / 100
        except (KeyError, IndexError):
            print("Fehler beim Parsen der FMP API Antwort, benutze Fallbackwert.")
            return 0.015

    else:
        raise ValueError("Unsupported source for risk-free rate")




# ======================== MACRO INDICATORS ============================

def get_macro_indicators_fmp() -> dict:
    """Fetch general US macroeconomic indicators from FMP."""
    url = f"https://financialmodelingprep.com/api/v4/us-economic-indicators/?apikey={FMP_API_KEY}"
    return requests.get(url).json()

def get_beta_fmp(ticker: str) -> float:
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
    response = requests.get(url).json()
    try:
        return float(response[0]['beta'])
    except (KeyError, IndexError):
        return None


def get_earnings_calendar(ticker: str) -> dict:
    url = f"https://financialmodelingprep.com/api/v3/earning_calendar/{ticker}?apikey={FMP_API_KEY}"
    return requests.get(url).json()


def get_insider_trading_fmp(ticker: str) -> dict:
    url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&apikey={FMP_API_KEY}"
    return requests.get(url).json()


def get_technical_indicator(ticker: str, indicator: str = "SMA", interval: str = "daily", time_period: int = 20, series_type: str = "close") -> dict:
    url = f"https://www.alphavantage.co/query?function={indicator}&symbol={ticker}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey={ALPHA_VANTAGE_API_KEY}"
    return requests.get(url).json()

def get_stock_returns(ticker: str, start: str = "2020-01-01", end: str = None) -> pd.Series:
    """Returns daily percentage returns for a given ticker."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    return get_daily_returns(ticker, start, end)
