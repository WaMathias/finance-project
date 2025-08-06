import yfinance as yf
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