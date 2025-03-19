import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from binance_data_fetcher import BinanceDataFetcher
from datetime import datetime, timedelta
from binance.client import Client

import pandas as pd
import numpy as np


def detect_pivot_volume_order_blocks(
    df,
    length=5,
    bull_ext_last=3,
    bear_ext_last=10,
    mitigation_method='Close',
    volume_lookback=20,
    atr_period=14,
    min_height_multiplier=0.5,
    use_market_structure=True,
    strength_threshold=70
):
    """
    Detect Order Blocks and only include those that are strong enough to potentially cause rejection.

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
        strength_threshold (int): Minimum strength score (0-100) for an OB to be included.

    Returns:
        tuple: (Updated DataFrame with OB signals, list of active order blocks).
    """

    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Check if volume data is available in the DataFrame
    has_volume = 'volume' in df.columns

    # Calculate volume moving average if volume data exists
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        # Fill NaN values with the first volume value or 0 if DataFrame is empty
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    # Calculate True Range (TR) and Average True Range (ATR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['atr'] = df['atr'].fillna(df['tr'].mean())  # Fill NaN with mean TR

    # Calculate rolling highs and lows for pivot detection
    df['upper'] = df['high'].rolling(length).max()
    df['lower'] = df['low'].rolling(length).min()

    # Add market structure (os) if enabled
    if use_market_structure:
        df['os'] = 0  # 0 for bearish, 1 for bullish
        for i in range(length, len(df)):
            high_prev = df['high'].iloc[i - length]
            low_prev = df['low'].iloc[i - length]
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]
            if high_prev > upper:
                df.loc[df.index[i], 'os'] = 0  # Bearish structure
            elif low_prev < lower:
                df.loc[df.index[i], 'os'] = 1  # Bullish structure
            else:
                # Retain previous state
                df.loc[df.index[i], 'os'] = df['os'].iloc[i - 1]

    # Identify pivot volume highs if volume data is available
    df['phv'] = False
    if has_volume:
        for k in range(length, len(df) - length):
            if (df['volume'].iloc[k] > df['volume'].iloc[k - length:k].max() and
                    df['volume'].iloc[k] > df['volume'].iloc[k + 1:k + length + 1].max()):
                df.at[df.index[k], 'phv'] = True

    # Define mitigation targets based on the specified method
    if mitigation_method == 'Close':
        df['target_bull'] = df['close'].rolling(length).min()
        df['target_bear'] = df['close'].rolling(length).max()
    else:  # 'Wick'
        df['target_bull'] = df['low'].rolling(length).min()
        df['target_bear'] = df['high'].rolling(length).max()

    # Initialize DataFrame columns for OB signals and mitigation flags
    df['bull_ob'] = np.nan
    df['bear_ob'] = np.nan
    df['mitigated_bull'] = False
    df['mitigated_bear'] = False

    # Initialize lists to store bullish and bearish order blocks
    bull_obs = []
    bear_obs = []

    # Main loop to detect OBs and filter based on strength
    for i in range(2 * length, len(df)):
        current_time = df.index[i]

        # Check for pivot volume high at bar k (i - length)
        if has_volume and df['phv'].iloc[i - length]:
            k = i - length  # Index of the pivot bar
            # Determine OB direction based on market structure or price movement
            direction = 'bullish' if df['os'].iloc[i] == 1 else 'bearish' if use_market_structure else (
                'bullish' if df['close'].iloc[i] > df['close'].iloc[i -
                                                                    length] else 'bearish'
            )

            if direction == 'bullish':
                # Define bullish OB coordinates
                top = (df['high'].iloc[k] + df['low'].iloc[k]) / \
                    2  # Midpoint (hl2)
                bottom = df['low'].iloc[k]
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    top = bottom + min_height  # Adjust top to ensure minimum height

                # Create OB dictionary
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
                    'height_atr_ratio': (top - bottom) / df['atr'].iloc[k],
                    'volume': 0
                }

                # Calculate strength if volume is available
                if has_volume:
                    vol = sum(df.at[df.index[i], 'volume']
                              for i in [k, k+1] if i < len(df))
                    ob['volume'] = vol
                    volume_k = df['volume'].iloc[k]
                    volume_ma_k = df['volume_ma'].iloc[k]
                    volume_ratio = volume_k / volume_ma_k if volume_ma_k > 0 else 1
                    height_ratio = ob['height'] / \
                        ob['atr'] if ob['atr'] > 0 else 1
                    # Strength is a combination of volume and height ratios
                    volume_strength = min(volume_ratio * 50, 50)
                    height_strength = min(height_ratio * 50, 50)
                    ob['strength'] = int(volume_strength + height_strength)

                    # Only include OB if its strength meets the threshold
                    if ob['strength'] >= strength_threshold:
                        bull_obs.insert(0, ob)  # Add to the front of the list
                        # Mark in DataFrame
                        df.at[current_time, 'bull_ob'] = bottom

            else:  # Bearish OB
                # Define bearish OB coordinates
                top = df['high'].iloc[k]
                bottom = (df['high'].iloc[k] + df['low'].iloc[k]
                          ) / 2  # Midpoint (hl2)
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    bottom = top - min_height  # Adjust bottom to ensure minimum height

                # Create OB dictionary
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
                    'height_atr_ratio': (top - bottom) / df['atr'].iloc[k],
                    'volume': 0
                }

                # Calculate strength if volume is available
                if has_volume:
                    vol = sum(df.at[df.index[i], 'volume']
                              for i in [k, k+1] if i < len(df))
                    ob['volume'] = vol
                    volume_k = df['volume'].iloc[k]
                    volume_ma_k = df['volume_ma'].iloc[k]
                    volume_ratio = volume_k / volume_ma_k if volume_ma_k > 0 else 1
                    height_ratio = ob['height'] / \
                        ob['atr'] if ob['atr'] > 0 else 1
                    # Strength is a combination of volume and height ratios
                    volume_strength = min(volume_ratio * 50, 50)
                    height_strength = min(height_ratio * 50, 50)
                    ob['strength'] = int(volume_strength + height_strength)

                    # Only include OB if its strength meets the threshold
                    if ob['strength'] >= strength_threshold:
                        # Add to the front of the list
                        bear_obs.insert(0, ob)
                        # Mark in DataFrame
                        df.at[current_time, 'bear_ob'] = top

        # Check for OB mitigation
        target_bull = df['target_bull'].iloc[i]
        target_bear = df['target_bear'].iloc[i]

        # Mitigate bullish OBs if price drops below the bottom
        for ob in bull_obs[:]:
            if target_bull < ob['bottom']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bull_obs.remove(ob)
                df.at[current_time, 'mitigated_bull'] = True

        # Mitigate bearish OBs if price rises above the top
        for ob in bear_obs[:]:
            if target_bear > ob['top']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bear_obs.remove(ob)
                df.at[current_time, 'mitigated_bear'] = True

        # Limit the number of active OBs in the lists
        bull_obs = bull_obs[:bull_ext_last]
        bear_obs = bear_obs[:bear_ext_last]

    # Return the updated DataFrame and combined list of active OBs
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
    client = Client()
    fetchData = BinanceDataFetcher(client=client)
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
