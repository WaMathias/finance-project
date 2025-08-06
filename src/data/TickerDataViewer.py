import yfinance as yf
import matplotlib.pyplot as plt

class TickerDataViewer:
    def __init__(self, tickers):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers

    def plot_price_chart(self, start_date, end_date):
        data = yf.download(self.tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(data[ticker], label=f'Close Price {ticker}')
            mean_price = data[ticker].mean()
            plt.axhline(mean_price, color='red', linestyle='--', label=f'Mean {ticker}: {mean_price:.2f}')

        plt.title(f"Close Prices for {', '.join(self.tickers)} from {start_date} to {end_date}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
