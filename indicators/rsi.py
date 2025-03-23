from typing import Union
import pandas as pd


def calculate_rsi(data: Union[pd.DataFrame, pd.Series], rsi_length: int = 14, ma_type: str = "SMA") -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)

    Parameters:
    - data: DataFrame with OHLCV data or Series of values
    - periods: RSI period (default 14)

    Returns:
    - RSI values as Series
    """
    # Handle both DataFrame and Series inputs
    if isinstance(data, pd.DataFrame):
        series = data['close']
    else:
        series = data

    delta = series.diff()
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the smoothed average gains and losses
    if ma_type == "SMA":
        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()
    elif ma_type == "EMA":
        avg_gain = gain.ewm(span=rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(span=rsi_length, adjust=False).mean()
    else:
        raise ValueError("Invalid moving average type. Use 'SMA' or 'EMA'.")
    for i in range(rsi_length, len(data)):

        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] *
                            (rsi_length - 1) + gain.iloc[i]) / rsi_length

        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] *
                            (rsi_length - 1) + loss.iloc[i]) / rsi_length

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """Calculate the MACD and Signal line."""
    exp1 = data['close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal

    macd_current = macd.iloc[-1]
    macd_prev = macd.iloc[-2]
    signal_current = signal.iloc[-1]
    signal_prev = signal.iloc[-2]
    hist_current = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2]

    macd_cross_up = macd_current > signal_current and macd_prev <= signal_prev
    macd_cross_down = macd_current < signal_current and macd_prev >= signal_prev
    hist_pos_to_neg = hist_current < 0 and hist_prev >= 0
    hist_neg_to_pos = hist_current >= 0 and hist_prev < 0

    signals = {
        "buy_signal": macd_cross_up,
        "sell_signal": macd_cross_down,
        "hist_confirmation_buy": hist_neg_to_pos,
        "hist_confirmation_sell": hist_pos_to_neg,
        "macd_direction": "UP" if macd_current > macd_prev else "DOWN",
        "histogram_direction": "UP" if hist_current > hist_prev else "DOWN",
        "macd": macd,
        "signal": signal,
        "histogram": histogram
    }
    return signals
