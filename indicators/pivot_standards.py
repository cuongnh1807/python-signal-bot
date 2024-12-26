import pandas as pd


def pivot_points(self, ohlc: pd.DataFrame, pivot_type='Traditional', pivot_timeframe='Auto',
                 max_historical_pivots=15, use_daily_based=True) -> pd.DataFrame:
    """
       Calculate pivot points similar to TradingView

       Parameters:
       - pivot_type: str
           'Traditional', 'Fibonacci', 'Woodie', 'Classic', 'DM', 'Camarilla'
       - pivot_timeframe: str
           'Auto', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly', 'Biyearly', 'Triyearly', 'Quinquennially', 'Decennially'
       - max_historical_pivots: int
           Maximum number of historical pivot sets to display
       - use_daily_based: bool
           Use daily-based calculations instead of intraday
       """

    # Determine the pivot timeframe
    if pivot_timeframe == 'Auto':
        if ohlc.index.freq and 'min' in str(ohlc.index.freq):
            pivot_timeframe = '1D' if ohlc.index.freq.n <= 15 else '1W'
        else:
            pivot_timeframe = '1D'

    if pivot_timeframe == '1D':
        resampled = ohlc.resample('D').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    elif pivot_timeframe == '1W':
        resampled = ohlc.resample('W').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    elif pivot_timeframe == '1M':
        resampled = ohlc.resample('M').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    else:
        resampled = ohlc  # Default to original data if no valid timeframe

    pivot_levels = []
    for i in range(max_historical_pivots):
        if len(resampled) < i + 1:
            break

        high = resampled['high'].iloc[-(i + 1)]
        low = resampled['low'].iloc[-(i + 1)]
        close = resampled['close'].iloc[-(i + 1)]
        open_price = resampled['open'].iloc[-(i + 1)]

        if pivot_type == 'Traditional':
            PP = (high + low + close) / 3
            R1 = (2 * PP) - low
            S1 = (2 * PP) - high
            R2 = PP + (high - low)
            S2 = PP - (high - low)
            R3 = high + 2 * (PP - low)
            S3 = low - 2 * (high - PP)
            levels = {'PP': PP, 'R1': R1, 'S1': S1,
                      'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
        elif pivot_type == 'Fibonacci':
            PP = (high + low + close) / 3
            R1 = PP + 0.382 * (high - low)
            S1 = PP - 0.382 * (high - low)
            R2 = PP + 0.618 * (high - low)
            S2 = PP - 0.618 * (high - low)
            R3 = PP + (high - low)
            S3 = PP - (high - low)
            levels = {'PP': PP, 'R1': R1, 'S1': S1,
                      'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
