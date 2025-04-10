import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from binance_data_fetcher import BinanceDataFetcher
from datetime import datetime, timedelta
from binance.client import Client

import numpy as np
import pandas as pd

from indicators.candles import should_keep_ob
from indicators.rsi import calculate_macd, calculate_rsi
from helpers.price import merge_overlapping_order_blocks


def detect_pivot_volume_order_blocks(
    df,
    length=5,
    bull_ext_last=3,
    bear_ext_last=3,
    mitigation_method='Wick',
    volume_lookback=20,
    atr_period=14,
    min_height_multiplier=0.5,
    use_market_structure=True,
    strength_threshold=70,
    use_should_keep_ob=True
):
    """
    Detect order blocks based on pivot volume strategy, similar to the TradingView indicator.

    Parameters:
        df (pd.DataFrame): OHLCV dataframe
        length (int): Volume pivot length (lookback period)
        bull_ext_last (int): Number of bullish OBs to keep
        bear_ext_last (int): Number of bearish OBs to keep
        mitigation_method (str): 'Wick' or 'Close' for determining mitigation
        volume_lookback (int): Period for volume moving average
        atr_period (int): Period for ATR calculation
        min_height_multiplier (float): Minimum height as ATR multiple
        use_market_structure (bool): Whether to use market structure for OB direction
        strength_threshold (int): Minimum strength required for an OB

    Returns:
        tuple: (df with added columns, list of order blocks)
    """
    df = df.copy()

    has_volume = 'volume' in df.columns

    # Base indicators
    df['rsi'] = calculate_rsi(df)

    macd_info = calculate_macd(df)
    df['macd'] = macd_info['macd']
    df['macd_signal'] = macd_info['signal']
    df['macd_hist'] = macd_info['histogram']

    # Volume indicators
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    # ATR for sizing
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['atr'] = df['atr'].fillna(df['tr'].mean())

    # Calculate highest and lowest prices over the length period
    # This aligns with TradingView's upper/lower calculation
    df['upper'] = df['high'].rolling(length).max()
    df['lower'] = df['low'].rolling(length).min()

    # Initialize order state (os)
    df['os'] = 0  # 0 = bearish context, 1 = bullish context

    # Market structure determination based on TradingView logic
    if use_market_structure:
        for i in range(length, len(df)):
            if i < length:
                continue

            high_prev = df['high'].iloc[i - length]
            low_prev = df['low'].iloc[i - length]
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]

            # Logic directly from TradingView: os := high[length] > upper ? 0 : low[length] < lower ? 1 : os[1]
            if high_prev > upper:
                df.loc[df.index[i], 'os'] = 0  # Bearish context
            elif low_prev < lower:
                df.loc[df.index[i], 'os'] = 1  # Bullish context
            else:
                # Keep previous state
                df.loc[df.index[i], 'os'] = df['os'].iloc[i - 1]

    # Pivot high volume detection - this is a key part of the TradingView indicator
    df['phv'] = False
    if has_volume:
        for k in range(length, len(df) - length):
            # Check if volume at position k is the highest in the window [k-length:k+length]
            # This is equivalent to TradingView's: phv = ta.pivothigh(volume, length, length)
            volume_before = df['volume'].iloc[k -
                                              length:k].max() if k >= length else 0
            volume_after = df['volume'].iloc[k + 1:k +
                                             length + 1].max() if k + length < len(df) else 0

            if df['volume'].iloc[k] > volume_before and df['volume'].iloc[k] > volume_after:
                df.at[df.index[k], 'phv'] = True

    # Set mitigation targets based on method
    if mitigation_method == 'Close':
        df['target_bull'] = df['close'].rolling(length).min()
        df['target_bear'] = df['close'].rolling(length).max()
    else:  # 'Wick'
        # TradingView uses lower for bullish targets
        df['target_bull'] = df['lower']
        # TradingView uses upper for bearish targets
        df['target_bear'] = df['upper']

    # Initialize OB tracking
    df['bull_ob'] = np.nan
    df['bear_ob'] = np.nan
    df['mitigated_bull'] = False
    df['mitigated_bear'] = False

    bull_obs = []
    bear_obs = []

    # Main order block detection and tracking loop
    for i in range(2 * length, len(df)):
        current_time = df.index[i]

        # Order block detection based on pivot high volume and market structure
        if has_volume and df['phv'].iloc[i - length]:
            k = i - length  # Position of the potential order block

            # Direction determination based on order state (os)
            direction = 'bullish' if df['os'].iloc[i] == 1 else 'bearish'

            if direction == 'bullish':
                # Bullish OB: mid to low of candle (TradingView: hl2[length], low[length])
                top = (df['high'].iloc[k] + df['low'].iloc[k]) / 2  # hl2
                bottom = df['low'].iloc[k]

                # Ensure minimum height
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    top = bottom + min_height

                # Create order block object
                ob = {
                    'index': k,
                    'direction': 1,
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

                # Add volume-related metrics
                if has_volume:
                    valid_indices = [i for i in [k, k + 1] if i < len(df)]
                    vol = sum(df.at[df.index[idx], 'volume'] for idx in valid_indices) / \
                        len(valid_indices) if valid_indices else 0

                    ob['volume'] = vol
                    volume_k = df['volume'].iloc[k]
                    volume_ma = df['volume_ma'].iloc[len(df) - 1]
                    volume_ratio = volume_k / volume_ma if volume_ma > 0 else 1
                    height_ratio = ob['height'] / \
                        ob['atr'] if ob['atr'] > 0 else 1

                    # Count historical OBs in similar price range
                    ob_count = 0
                    recent_count = 0
                    ob_price_range = ob['height'] * 1.5

                    for j in range(max(0, k - 500), k):
                        if j < len(df) and df['phv'].iloc[j]:
                            avg_price_j = (
                                df['high'].iloc[j] + df['low'].iloc[j]) / 2
                            if abs(ob['avg'] - avg_price_j) <= ob_price_range:
                                ob_count += 1
                                if j >= max(0, k - 100):
                                    recent_count += 1

                    ob['historical_count'] = ob_count
                    ob['recent_count'] = recent_count

                    # Calculate OB strength based on volume, height, and historical presence
                    if recent_count >= 3:
                        historical_strength = max(
                            5, 15 - (recent_count - 2) * 5)
                    else:
                        historical_strength = min(ob_count * 4, 25)

                    volume_strength = min(volume_ratio * 40, 40)
                    height_strength = min(height_ratio * 35, 35)
                    ob['strength'] = int(
                        volume_strength + height_strength + historical_strength)

                    # Add to OB list if strong enough and passes additional filtering
                    if ob['strength'] >= strength_threshold:
                        keep_ob, result = should_keep_ob(
                            df, ob, len(df)-1, use_should_keep_ob)
                        if keep_ob:
                            # Add score and quality info to the order block
                            ob['score'] = result["final_score"]
                            ob['setup_quality'] = result["setup_quality"]
                            ob['warnings'] = result["warnings"]
                            ob['entry_quality'] = result.get(
                                "entry_quality", "Unknown")
                            bull_obs.insert(0, ob)
                            df.at[current_time, 'bull_ob'] = bottom

            else:  # Bearish OB
                # Bearish OB: high to mid of candle (TradingView: high[length], hl2[length])
                top = df['high'].iloc[k]
                bottom = (df['high'].iloc[k] + df['low'].iloc[k]) / 2  # hl2

                # Ensure minimum height
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    bottom = top - min_height

                # Create order block object
                ob = {
                    'index': k,
                    'direction': -1,
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

                # Add volume-related metrics
                if has_volume:
                    valid_indices = [i for i in [k, k + 1] if i < len(df)]
                    vol = sum(df.at[df.index[idx], 'volume'] for idx in valid_indices) / \
                        len(valid_indices) if valid_indices else 0

                    ob['volume'] = vol
                    volume_k = df['volume'].iloc[k]
                    volume_ma = df['volume_ma'].iloc[len(df) - 1]
                    volume_ratio = volume_k / volume_ma if volume_ma > 0 else 1
                    height_ratio = ob['height'] / \
                        ob['atr'] if ob['atr'] > 0 else 1

                    # Count historical OBs in similar price range
                    ob_count = 0
                    recent_count = 0
                    ob_price_range = ob['height'] * 1.5

                    for j in range(max(0, k - 500), k):
                        if j < len(df) and df['phv'].iloc[j]:
                            avg_price_j = (
                                df['high'].iloc[j] + df['low'].iloc[j]) / 2
                            if abs(ob['avg'] - avg_price_j) <= ob_price_range:
                                ob_count += 1
                                if j >= max(0, k - 100):
                                    recent_count += 1

                    ob['historical_count'] = ob_count
                    ob['recent_count'] = recent_count

                    # Calculate OB strength based on volume, height, and historical presence
                    if recent_count >= 3:
                        historical_strength = max(
                            5, 15 - (recent_count - 2) * 5)
                    else:
                        historical_strength = min(ob_count * 4, 20)

                    volume_strength = min(volume_ratio * 40, 40)
                    height_strength = min(height_ratio * 40, 40)
                    ob['strength'] = int(
                        volume_strength + height_strength + historical_strength)

                    # Add to OB list if strong enough and passes additional filtering
                    if ob['strength'] >= strength_threshold:
                        keep_ob, result = should_keep_ob(
                            df, ob, len(df)-1, use_should_keep_ob)
                        if keep_ob:
                            # Add score and quality info to the order block
                            ob['score'] = result["final_score"]
                            ob['setup_quality'] = result["setup_quality"]
                            ob['warnings'] = result["warnings"]
                            ob['entry_quality'] = result.get(
                                "entry_quality", "Unknown")
                            bear_obs.insert(0, ob)
                            df.at[current_time, 'bear_ob'] = top

        # Check for OB mitigation based on price movement
        # This follows TradingView's removal logic
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

        # Limit the number of OBs and merge overlapping ones
        bull_obs = bull_obs[:bull_ext_last]
        bear_obs = bear_obs[:bear_ext_last]
        bull_obs = merge_overlapping_order_blocks(bull_obs, 0.5)
        bear_obs = merge_overlapping_order_blocks(bear_obs, 0.5)

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
    parser.add_argument(
        '--length', type=int, default=5, help='Volume pivot length')
    parser.add_argument(
        '--bull_ext_last', type=int, default=5, help='Number of bullish OBs to keep')
    parser.add_argument(
        '--bear_ext_last', type=int, default=5, help='Number of bearish OBs to keep')
    parser.add_argument(
        '--mitigation_method', type=str, default='Wick', choices=['Wick', 'Close'],
        help='Method to determine when OBs are mitigated')

    parser.add_argument(
        '--use_should_keep_ob', type=str, default='True', help='Whether to use should_keep_ob for OB direction')

    args = parser.parse_args()
    client = Client()
    fetchData = BinanceDataFetcher(client=client)
    start_time = datetime.now() - timedelta(days=7)
    data = fetchData.get_historical_klines(
        args.symbol, interval=args.interval, start_time=start_time)
    print(args.use_should_keep_ob)
    df, orders = detect_pivot_volume_order_blocks(data,
                                                  length=args.length,
                                                  bull_ext_last=args.bull_ext_last,
                                                  bear_ext_last=args.bear_ext_last,
                                                  mitigation_method=args.mitigation_method,
                                                  atr_period=14,
                                                  min_height_multiplier=0.5,
                                                  use_should_keep_ob=True if args.use_should_keep_ob == 'True' else False
                                                  )

    # Separate order blocks by direction
    bullish_obs = [ob for ob in orders if ob['direction'] == 1]
    bearish_obs = [ob for ob in orders if ob['direction'] == -1]

    for ob in bullish_obs:
        print(
            f"Bullish start time: {ob['left_time']}, top: {ob['top']}, bottom: {ob['bottom']}, avg: {ob['avg']}, height: {ob['height']}, volume: {ob['volume']}, strength: {ob['strength']}")
    for ob in bearish_obs:
        print(
            f"Bearish start time: {ob['left_time']}, top: {ob['top']}, bottom: {ob['bottom']}, avg: {ob['avg']}, height: {ob['height']}, volume: {ob['volume']}, strength: {ob['strength']}")
    plot_order_blocks(df, bullish_obs, bearish_obs)
