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
        print(f"[DEBUG] Input types: stock_returns={type(stock_returns)}, market_returns={type(market_returns)}")
        print(
            f"[DEBUG] Input shapes: stock_returns={getattr(stock_returns, 'shape', 'no shape')}, market_returns={getattr(market_returns, 'shape', 'no shape')}")
        if hasattr(stock_returns, '__len__'):
            print(
                f"[DEBUG] Ursprüngliche Längen: stock_returns={len(stock_returns)}, market_returns={len(market_returns)}")

        if hasattr(stock_returns, 'index') and hasattr(market_returns, 'index'):
            print(f"[DEBUG] Stock returns index range: {stock_returns.index.min()} to {stock_returns.index.max()}")
            print(f"[DEBUG] Market returns index range: {market_returns.index.min()} to {market_returns.index.max()}")

    if not isinstance(stock_returns, pd.Series):
        raise TypeError(f"stock_returns muss eine pandas Series sein, bekommen: {type(stock_returns)}")
    if not isinstance(market_returns, pd.Series):
        raise TypeError(f"market_returns muss eine pandas Series sein, bekommen: {type(market_returns)}")

    if len(stock_returns) == 0 or len(market_returns) == 0:
        raise ValueError("Eine der Input-Series ist leer!")

    stock_aligned, market_aligned = stock_returns.align(market_returns, join='inner')

    if debug:
        print(f"[DEBUG] Nach Alignment: stock={len(stock_aligned)}, market={len(market_aligned)}")
        if len(stock_aligned) > 0:
            print(f"[DEBUG] Aligned index range: {stock_aligned.index.min()} to {stock_aligned.index.max()}")

    if len(stock_aligned) == 0:
        raise ValueError("Nach Alignment keine gemeinsamen Datenpunkte gefunden!")

    daily_risk_free = risk_free_rate / 252

    excess_stock = stock_aligned - daily_risk_free
    excess_market = market_aligned - daily_risk_free

    if debug:
        print(
            f"[DEBUG] Excess returns berechnet. NaN counts: stock={excess_stock.isna().sum()}, market={excess_market.isna().sum()}")

    try:
        combined_data = pd.concat([excess_stock, excess_market], axis=1, keys=['excess_stock', 'excess_market'])
        valid_data = combined_data.dropna()

        if debug:
            print(f"[DEBUG] Combined data shape: {combined_data.shape}")
            print(f"[DEBUG] Valid data shape nach dropna: {valid_data.shape}")

    except Exception as e:
        if debug:
            print(f"[DEBUG] Fehler beim Kombinieren der Daten: {e}")
        raise ValueError(f"Fehler beim Verarbeiten der Excess Returns: {e}")

    if len(valid_data) == 0:
        if debug:
            print("[DEBUG] Keine gültigen Datenpaare nach NaN-Filterung!")
            print(f"[DEBUG] Original stock_returns hat {stock_returns.isna().sum()} NaNs")
            print(f"[DEBUG] Original market_returns hat {market_returns.isna().sum()} NaNs")
        raise ValueError("Nach Filterung von NaNs keine Daten mehr übrig!")

    excess_stock_final = valid_data['excess_stock']
    excess_market_final = valid_data['excess_market']

    if debug:
        print(f"[DEBUG] Finale Datenlänge: {len(excess_stock_final)}")

    # Regression (excess_stock ~ excess_market)
    X = excess_market_final.values.reshape(-1, 1)
    y = excess_stock_final.values

    model = LinearRegression()
    model.fit(X, y)

    beta = model.coef_[0]
    alpha = model.intercept_
    r_squared = model.score(X, y)

    # Erwartete Rendite nach CAPM (auf Jahresbasis)
    expected_return = risk_free_rate + beta * (market_aligned.mean() * 252 - risk_free_rate)

    if debug:
        print(
            f"[DEBUG] Beta: {beta:.4f}, Alpha: {alpha:.4f}, R²: {r_squared:.4f}, Erwartete Rendite: {expected_return:.4f}")

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
    if not isinstance(stock_returns, pd.Series) or not isinstance(market_returns, pd.Series):
        print(f"Warnung: Ungültige Datentypen für Plot von {ticker}")
        return

    if len(stock_returns) == 0 or len(market_returns) == 0:
        print(f"Warnung: Leere Daten für Plot von {ticker}")
        return

    stock_aligned, market_aligned = stock_returns.align(market_returns, join='inner')

    if len(stock_aligned) == 0:
        print(f"Warnung: Keine gemeinsamen Datenpunkte für Plot von {ticker}")
        return

    daily_risk_free = risk_free_rate / 252

    excess_stock = stock_aligned - daily_risk_free
    excess_market = market_aligned - daily_risk_free

    try:
        combined_data = pd.concat([excess_stock, excess_market], axis=1, keys=['excess_stock', 'excess_market'])
        valid_data = combined_data.dropna()
    except Exception as e:
        print(f"Warnung: Fehler beim Verarbeiten der Daten für Plot von {ticker}: {e}")
        return

    if len(valid_data) == 0:
        print(f"Warnung: Keine gültigen Daten für Plot von {ticker}")
        return

    excess_stock_final = valid_data['excess_stock']
    excess_market_final = valid_data['excess_market']

    model = LinearRegression().fit(excess_market_final.values.reshape(-1, 1), excess_stock_final.values)
    y_pred = model.predict(excess_market_final.values.reshape(-1, 1))
    beta = model.coef_[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(excess_market_final, excess_stock_final, alpha=0.5, label="Excess Returns")
    plt.plot(excess_market_final, y_pred, color='red', label=f"CAPM Line (β ≈ {beta:.2f})")
    plt.title(f"CAMP Regression: {ticker} vs Market")
    plt.xlabel("Excess Market Return")
    plt.ylabel(f"Excess {ticker} Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()