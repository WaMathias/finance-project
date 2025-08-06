import yfinance as yf
import matplotlib.pyplot as plt

class TickerDataViewer:
    def __init__(self, tickers):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers

    def plot_chart(self, start_date, end_date):
        data = yf.download(self.tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(data[ticker], label=f'Schlusskurs {ticker}')
            mean_price = data[ticker].mean()
            plt.axhline(mean_price, color='red', linestyle='--', label=f'Durchschnitt {ticker}: {mean_price:.2f} EUR')

        plt.title(f"Schlusskurse von {', '.join(self.tickers)} von {start_date.date()} bis {end_date.date()}")
        plt.xlabel("Datum")
        plt.ylabel("Preis in EUR")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
