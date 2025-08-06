import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def compute_capm(stock_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float, debug=False) -> dict:
    """
    Berechnet CAPM-Kennzahlen (Beta, Alpha, R², Erwartete Rendite).
    Args:
        stock_returns: pd.Series mit Aktienrenditen (index = DatetimeIndex)
        market_returns: pd.Series mit Marktrenditen (index = DatetimeIndex)
        risk_free_rate: float, jährlicher risikofreier Zinssatz (z.B. 0.015 für 1.5%)
        debug: bool, ob Debug-Infos ausgegeben werden sollen

    Returns:
        dict mit Beta, Alpha, R² und Erwarteter Rendite
    """
    if debug:
        print(f"[DEBUG] Ursprüngliche Längen: stock_returns={len(stock_returns)}, market_returns={len(market_returns)}")

    # Risk-Free Rate als Series mit gleichem Index (angenommen tägliche Rendite = annual_rate / 252)
    risk_free_series = pd.Series(risk_free_rate / 252, index=stock_returns.index)

    # Aligniere alle Series auf gemeinsame Datumswerte (inner join)
    stock_returns, market_returns = stock_returns.align(market_returns, join='inner')
    stock_returns, risk_free_series = stock_returns.align(risk_free_series, join='inner', axis=0)

    if debug:
        print(f"[DEBUG] Gemeinsamer Index Länge: {len(stock_returns)}")

    # Excess Returns berechnen
    excess_stock = stock_returns - risk_free_series
    excess_market = market_returns - risk_free_series

    # NaN-Werte rausfiltern (z.B. Feiertage)
    valid_mask = excess_stock.notna() & excess_market.notna()
    excess_stock = excess_stock.loc[valid_mask]
    excess_market = excess_market.loc[valid_mask]

    if len(excess_stock) == 0 or len(excess_market) == 0:
        raise ValueError("Nach Filterung von NaNs keine Daten mehr übrig!")

    # Regression (excess_stock ~ excess_market)
    X = excess_market.values.reshape(-1, 1)
    y = excess_stock.values

    model = LinearRegression()
    model.fit(X, y)

    beta = model.coef_[0]
    alpha = model.intercept_
    r_squared = model.score(X, y)

    # Erwartete Rendite nach CAPM (auf Jahresbasis)
    expected_return = risk_free_rate + beta * (market_returns.mean() * 252 - risk_free_rate)

    if debug:
        print(f"[DEBUG] Beta: {beta:.4f}, Alpha: {alpha:.4f}, R²: {r_squared:.4f}, Erwartete Rendite: {expected_return:.4f}")

    return {
        "Beta": beta,
        "Alpha": alpha,
        "R²": r_squared,
        "Expected Return": expected_return
    }

def plot_capm(stock_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float, ticker: str):
    """
    Plottet die Regression der Überschussrenditen von Aktie vs Markt.
    """
    risk_free_series = pd.Series(risk_free_rate / 252, index=stock_returns.index)
    stock_returns, market_returns = stock_returns.align(market_returns, join='inner')
    stock_returns, risk_free_series = stock_returns.align(risk_free_series, join='inner')

    excess_stock = stock_returns - risk_free_series
    excess_market = market_returns - risk_free_series

    valid_mask = excess_stock.notna() & excess_market.notna()
    excess_stock = excess_stock.loc[valid_mask]
    excess_market = excess_market.loc[valid_mask]

    model = LinearRegression().fit(excess_market.values.reshape(-1, 1), excess_stock.values)
    y_pred = model.predict(excess_market.values.reshape(-1, 1))
    beta = model.coef_[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(excess_market, excess_stock, alpha=0.5, label="Excess Returns")
    plt.plot(excess_market, y_pred, color='red', label=f"CAPM Line (β ≈ {beta:.2f})")
    plt.title(f"CAPM Regression: {ticker} vs Market")
    plt.xlabel("Excess Market Return")
    plt.ylabel(f"Excess {ticker} Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()