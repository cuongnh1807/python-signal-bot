import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from binance_data_fetcher import BinanceDataFetcher
from datetime import datetime, timedelta


def detect_pivot_volume_order_blocks(
    df,
    length=5,
    bull_ext_last=3,
    bear_ext_last=10,
    mitigation_method='Close',
    volume_lookback=20,
    atr_period=14,
    min_height_multiplier=0.7,
    use_market_structure=True
):
    """
    Detect Order Blocks in OHLCV data using pivot volume and market structure.

    Parameters:
        df (pd.DataFrame): DataFrame containing OHLCV data with datetime index.
        length (int): Number of bars for pivot detection.
        bull_ext_last (int): Number of recent bullish OBs to track.
        bear_ext_last (int): Number of recent bearish OBs to track.
        mitigation_method (str): 'Wick' or 'Close' price for mitigation.
        volume_lookback (int): Lookback period for volume moving average.
        atr_period (int): Period for ATR calculation.
        min_height_multiplier (float): Minimum height as multiplier of ATR.
        use_market_structure (bool): Use market structure for OB direction.

    Returns:
        tuple: (Updated DataFrame with OB signals, list of active order blocks).
    """

    df = df.copy()

    # Check if volume data is available
    has_volume = 'volume' in df.columns

    # Calculate volume moving average if volume data exists
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['atr'] = df['atr'].fillna(df['tr'].mean())

    # Calculate upper and lower bounds
    df['upper'] = df['high'].rolling(length).max()
    df['lower'] = df['low'].rolling(length).min()

    # Add market structure (os) if enabled
    if use_market_structure:
        df['os'] = 0  # Initialize
        for i in range(length, len(df)):
            high_prev = df['high'].iloc[i - length]
            low_prev = df['low'].iloc[i - length]
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]
            if high_prev > upper:
                df.loc[df.index[i], 'os'] = 0  # Bearish
            elif low_prev < lower:
                df.loc[df.index[i], 'os'] = 1  # Bullish
            else:
                df.loc[df.index[i], 'os'] = df['os'].iloc[i - 1]

    # Precompute pivot volume highs
    df['phv'] = False
    if has_volume:
        for k in range(length, len(df) - length):
            if (df['volume'].iloc[k] > df['volume'].iloc[k - length:k].max() and
                    df['volume'].iloc[k] > df['volume'].iloc[k + 1:k + length + 1].max()):
                df.at[df.index[k], 'phv'] = True

    # Calculate mitigation targets
    if mitigation_method == 'Close':
        df['target_bull'] = df['close'].rolling(length).min()
        df['target_bear'] = df['close'].rolling(length).max()
    else:  # 'Wick'
        df['target_bull'] = df['low'].rolling(length).min()
        df['target_bear'] = df['high'].rolling(length).max()

    # Initialize columns for signals
    df['bull_ob'] = np.nan
    df['bear_ob'] = np.nan
    df['mitigated_bull'] = False
    df['mitigated_bear'] = False

    # Initialize lists for order blocks
    bull_obs = []
    bear_obs = []

    # Detect order blocks
    for i in range(2 * length, len(df)):
        current_time = df.index[i]

        # Check for order blocks if volume pivot high exists at i - length
        if has_volume and df['phv'].iloc[i - length]:
            k = i - length  # Bar where pivot high occurred
            direction = 'bullish' if df['os'].iloc[i] == 1 else 'bearish' if use_market_structure else (
                'bullish' if df['close'].iloc[i] > df['close'].iloc[i -
                                                                    length] else 'bearish'
            )

            if direction == 'bullish':
                # Bullish OB coordinates per Pine Script
                top = (df['high'].iloc[k] + df['low'].iloc[k]) / 2  # hl2
                bottom = df['low'].iloc[k]
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    top = bottom + min_height

                ob = {
                    'direction': 'bullish',
                    'left_time': df.index[k],
                    'top': top,
                    'bottom': bottom,
                    'avg': (top + bottom) / 2,
                    'height': top - bottom,
                    'mitigated': False,
                    'mitigated_time': None,
                    'atr': df['atr'].iloc[k],
                    'height_atr_ratio': (top - bottom) / df['atr'].iloc[k]
                }
                if has_volume:
                    ob['volume'] = df['volume'].iloc[k]
                    if df['volume_ma'].iloc[k] > 0:
                        ob['strength'] = min(
                            int((ob['volume'] / df['volume_ma'].iloc[k]) * 100), 100)
                bull_obs.insert(0, ob)
                df.at[current_time, 'bull_ob'] = bottom

            else:  # Bearish
                # Bearish OB coordinates per Pine Script
                top = df['high'].iloc[k]
                bottom = (df['high'].iloc[k] + df['low'].iloc[k]) / 2  # hl2
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    bottom = top - min_height

                ob = {
                    'direction': 'bearish',
                    'left_time': df.index[k],
                    'top': top,
                    'bottom': bottom,
                    'avg': (top + bottom) / 2,
                    'height': top - bottom,
                    'mitigated': False,
                    'mitigated_time': None,
                    'atr': df['atr'].iloc[k],
                    'height_atr_ratio': (top - bottom) / df['atr'].iloc[k]
                }
                if has_volume:
                    ob['volume'] = df['volume'].iloc[k]
                    if df['volume_ma'].iloc[k] > 0:
                        ob['strength'] = min(
                            int((ob['volume'] / df['volume_ma'].iloc[k]) * 100), 100)
                bear_obs.insert(0, ob)
                df.at[current_time, 'bear_ob'] = top

        # Mitigation checks
        target_bull = df['target_bull'].iloc[i]
        target_bear = df['target_bear'].iloc[i]

        # Mitigate bullish OBs
        for ob in bull_obs[:]:
            if target_bull < ob['bottom']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bull_obs.remove(ob)
                df.at[current_time, 'mitigated_bull'] = True

        # Mitigate bearish OBs
        for ob in bear_obs[:]:
            if target_bear > ob['top']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bear_obs.remove(ob)
                df.at[current_time, 'mitigated_bear'] = True

        # Limit the number of active OBs
        bull_obs = bull_obs[:bull_ext_last]
        bear_obs = bear_obs[:bear_ext_last]

    return df, bull_obs + bear_obs


def plot_order_blocks(df, all_bull_obs, all_bear_obs):
    """
    Visualize order blocks on price chart.

    Parameters:
        df (pd.DataFrame): OHLCV DataFrame
        all_bull_obs (list): Bullish order blocks data
        all_bear_obs (list): Bearish order blocks data
    """
    plt.figure(figsize=(14, 7))

    # Plot price data
    plt.plot(df.index, df['close'], label='Close Price', color='#2c3e50', lw=1)

    # Plot bullish order blocks
    for ob in all_bull_obs:
        start = ob['left_time']
        end = ob['mitigated_time'] or df.index[-1]
        plt.fill_betweenx([ob['bottom'], ob['top']],
                          start, end,
                          color='#169400', alpha=0.15, edgecolor='none')
        plt.hlines(ob['avg'], start, end,
                   colors='#169400', linestyles='dashed', linewidth=1, alpha=0.5)

    # Plot bearish order blocks
    for ob in all_bear_obs:
        start = ob['left_time']
        end = ob['mitigated_time'] or df.index[-1]
        plt.fill_betweenx([ob['bottom'], ob['top']],
                          start, end,
                          color='#ff1100', alpha=0.15, edgecolor='none')
        plt.hlines(ob['avg'], start, end,
                   colors='#ff1100', linestyles='dashed', linewidth=1, alpha=0.5)

    # Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.grid(alpha=0.2)
    plt.title('Order Block Detection')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Usage example:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-ticker trading bot')
    parser.add_argument(
        '--symbol', type=str, default='SOLUSDT', help='Symbol to fetch data')
    parser.add_argument(
        '--interval', type=str, default='15m', help='Interval to fetch data')
    args = parser.parse_args()
    fetchData = BinanceDataFetcher()
    start_time = datetime.now() - timedelta(days=7)
    data = fetchData.get_historical_klines(
        args.symbol, interval=args.interval, start_time=start_time)
    # df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close',
    #                                  'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # df.set_index('timestamp', inplace=True)
    df, orders = detect_pivot_volume_order_blocks(data, length=5,
                                                  bull_ext_last=5, bear_ext_last=10,
                                                  atr_period=14, min_height_multiplier=0.5)
    # Tách order blocks theo hướng
    bullish_obs = [ob for ob in orders if ob['direction'] == 'bullish']
    bearish_obs = [ob for ob in orders if ob['direction'] == 'bearish']
    print("Bullish Order Blocks:", bullish_obs)
    print("Bearish Order Blocks:", bearish_obs)
    plot_order_blocks(df, bullish_obs, bearish_obs)
