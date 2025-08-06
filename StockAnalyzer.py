import yfinance as yf

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.hist = self.stock.history(period="1y")
        self.returns = self.hist["Close"].pct_change().dropna()

    def analyze(self):
        print(f"\n📊 Analyse für: {self.ticker.upper()}")
        self.display_price_data()
        self.display_dividend_info()
        self.display_company_structure()
        self.display_valuation_metrics()
        self.display_fundamental_data()
        self.display_company_description()
        self.display_dividend_history()
        print("\n✅ Analyse abgeschlossen.")

    def display_price_data(self):
        volatility = self.returns.std() * (252 ** 0.5)  # annualisierte Volatilität
        print("\nKursentwicklung (letztes Jahr):")
        print(f"Aktueller Kurs: {self.hist['Close'][-1]:.2f}")
        print(f"52W Hoch: {self.hist['Close'].max():.2f}")
        print(f"52W Tief: {self.hist['Close'].min():.2f}")
        print(f"Jahresrendite: {self.returns.sum():.2%}")
        print(f"Volatilität (annualisiert): {volatility:.2%}")

    def display_dividend_info(self):
        info = self.stock.info
        print("\nDividenden:")
        print(f"Dividendenrendite: {info.get('dividendYield', 'n/a')}")
        print(f"Dividende je Aktie: {info.get('dividendRate', 'n/a')}")
        print(f"Ausschüttungsquote (Payout Ratio): {info.get('payoutRatio', 'n/a')}")

    def display_company_structure(self):
        info = self.stock.info
        print("\nUnternehmensstruktur:")
        print(f"Marktkapitalisierung: {info.get('marketCap', 'n/a')}")
        print(f"Anzahl Aktien im Umlauf: {info.get('sharesOutstanding', 'n/a')}")
        print(f"Beta: {info.get('beta', 'n/a')}")

    def display_valuation_metrics(self):
        info = self.stock.info
        print("\nBewertungskennzahlen:")
        print(f"KGV (PE Ratio): {info.get('trailingPE', 'n/a')}")
        print(f"KUV (Price/Sales): {info.get('priceToSalesTrailing12Months', 'n/a')}")
        print(f"KBV (Price/Book): {info.get('priceToBook', 'n/a')}")

    def display_fundamental_data(self):
        info = self.stock.info
        print("\nFundamentale Daten:")
        print(f"Umsatz (ttm): {info.get('totalRevenue', 'n/a')}")
        print(f"Gewinn (ttm): {info.get('netIncomeToCommon', 'n/a')}")
        print(f"EBITDA: {info.get('ebitda', 'n/a')}")
        print(f"Free Cashflow: {info.get('freeCashflow', 'n/a')}")
        print(f"Verschuldung: {info.get('totalDebt', 'n/a')}")
        print(f"Eigenkapitalrendite (ROE): {info.get('returnOnEquity', 'n/a')}")
        print(f"Gesamtkapitalrendite (ROA): {info.get('returnOnAssets', 'n/a')}")

    def display_company_description(self):
        info = self.stock.info
        print("\nBeschreibung:")
        print(info.get("longBusinessSummary", "Keine Beschreibung verfügbar."))

    def display_dividend_history(self):
        dividends = self.stock.dividends
        if not dividends.empty:
            print("\n📜 Dividendenhistorie:")
            print(dividends.tail(5))
