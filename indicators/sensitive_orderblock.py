import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from binance.client import Client

from binance_data_fetcher import BinanceDataFetcher
from indicators.candles import analyze_candle_volume
from indicators.rsi import calculate_macd
from helpers.price import detect_trend_from_ema, merge_overlapping_order_blocks
from helpers.candles import should_keep_ob


def detect_order_sensitive_blocks(df, sens=0.28, OBMitigationType="Close", buy_alert=False, sell_alert=False, volume_lookback=20, merge_threshold=0.7, max_blocks=20, atr_period=14, strength_threshold=70, use_should_keep_ob=True):
    """
    Detect bullish and bearish order blocks in a financial dataset based on Pine Script logic.
    Implements the Sonarlab Order Block detection algorithm from TradingView.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'open', 'high', 'low', 'close' and datetime index.
    - sens (float): Sensitivity for ROC crossover (default 0.28).
    - OBMitigationType (str): "Close" or "Wick" for mitigation logic.
    - buy_alert (bool): Enable buy alerts for bullish order blocks.
    - sell_alert (bool): Enable sell alerts for bearish order blocks.
    - volume_lookback (int): Lookback period for volume moving average.
    - merge_threshold (float): Threshold for merging overlapping order blocks (0-1).
    - max_blocks (int): Maximum number of order blocks to retain (like max_boxes_count in PineScript).
    - atr_period (int): Lookback period for ATR calculation.
    - strength_threshold (int): Minimum strength required for an order block.

    Returns:
    - list: Combined list of active order blocks (both bearish and bullish).
    """
    # Store original index before reset
    original_index = df.index.copy()

    # Ensure DataFrame has integer index for calculations

    analysis = detect_trend_from_ema(df, lookback=30)
    df = analysis['data']
    del analysis['data']
    df = df.reset_index(drop=True)

    # Calculate MACD for filtering
    macd_info = calculate_macd(df)

    df['macd'] = macd_info['macd']
    df['macd_signal'] = macd_info['signal']
    df['macd_hist'] = macd_info['histogram']

    # Calculate volume moving average if volume data exists
    has_volume = 'volume' in df.columns
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    # Calculate ATR for height strength evaluation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['atr'] = df['atr'].fillna(df['tr'].mean())

    # Calculate ROC (Rate of Change) - core signal for Sonarlab logic
    # This matches exactly the Pine Script: pc = (open - open[4]) / open[4] * 100
    df['pc'] = (df['open'] - df['open'].shift(4)) / df['open'].shift(4) * 100

    df['crossunder'] = (df['pc'].shift(1) > -sens) & (df['pc'] <= -sens)
    df['crossover'] = (df['pc'].shift(1) < sens) & (df['pc'] >= sens)

    # Store history of order blocks for historical_count calculation
    historical_obs = []

    # Initialize lists for active order blocks
    bearish_obs = []
    bullish_obs = []

    # Variables to track recent crossover/crossunder positions
    last_cross_bearish = None
    last_cross_bullish = None
    cross_index = 0

    # Process each bar
    for idx, row in df.iterrows():
        if idx < 4:  # Skip initial bars that don't have enough history
            continue

        # Keep track of the current cross_index (similar to Pine Script's cross_index)
        if row['crossunder'] or row['crossover']:
            prev_cross_index = cross_index
            cross_index = idx

            # Check if we should create a new order block based on minimum distance of 5 bars
            # This matches the Pine Script check: cross_index - cross_index[1] > 5
            if cross_index - prev_cross_index <= 5:
                continue

        # Bearish order block detection (after price momentum shift down)
        if row['crossunder']:
            if last_cross_bearish is None or (idx - last_cross_bearish) > 5:
                last_cross_bearish = idx

                # Look back for a green (bullish) candle to place the bearish order block
                # This matches Pine Script: for i = 4 to 15 by 1; if close[i] > open[i]; last_green := i; break
                for i in range(4, 20):
                    lookback_idx = idx - i
                    if lookback_idx < 0:
                        break

                    # Find bullish candles (close > open)
                    if df.loc[lookback_idx, 'close'] > df.loc[lookback_idx, 'open']:
                        # Create bearish order block on this candle
                        ob = {
                            'index': lookback_idx,
                            'direction': -1,
                            'left_time': original_index[lookback_idx],
                            # Use entire candle high as top
                            'top': df.loc[lookback_idx, 'high'],
                            # Use entire candle low as bottom
                            'bottom': df.loc[lookback_idx, 'low'],
                            'avg': (df.loc[lookback_idx, 'high'] + df.loc[lookback_idx, 'low']) / 2,
                            'height': df.loc[lookback_idx, 'high'] - df.loc[lookback_idx, 'low'],
                            'mitigated': False,
                            'mitigated_time': None,
                            'atr': df.loc[lookback_idx, 'atr'],
                            'height_atr_ratio': (df.loc[lookback_idx, 'high'] - df.loc[lookback_idx, 'low']) / df.loc[lookback_idx, 'atr'] if df.loc[lookback_idx, 'atr'] > 0 else 1,
                            'volume': df.loc[lookback_idx, 'volume'] if has_volume else 0,
                            'strength': calculate_ob_strength(df, lookback_idx, -1, historical_obs, has_volume)
                        }

                        # Add to list if strong enough
                        if ob['strength'] >= strength_threshold:
                            keep_ob, result = should_keep_ob(df, ob, len(
                                df)-1, use_should_keep_ob=use_should_keep_ob, analysis=analysis, strength_threshold=strength_threshold)
                            if keep_ob:
                                # Add score and quality info to the order block
                                ob['score'] = result["final_score"]
                                ob['setup_quality'] = result["setup_quality"]
                                ob['threshold'] = result["threshold"]
                                ob['warnings'] = result["warnings"]
                                ob['entry_quality'] = result.get(
                                    "entry_quality", "Unknown")
                                bearish_obs.append(ob)
                                historical_obs.append(ob)
                            break

        # Bullish order block detection (after price momentum shift up)
        if row['crossover']:
            if last_cross_bullish is None or (idx - last_cross_bullish) > 5:
                last_cross_bullish = idx

                # Look back for a red (bearish) candle to place the bullish order block
                # This matches Pine Script: for i = 4 to 15 by 1; if close[i] < open[i]; last_red := i; break
                for i in range(4, 16):
                    lookback_idx = idx - i
                    if lookback_idx < 0:
                        break

                    # Find bearish candles (close < open)
                    if df.loc[lookback_idx, 'close'] < df.loc[lookback_idx, 'open']:
                        # Create bullish order block on this candle
                        ob = {
                            'index': lookback_idx,
                            'direction': 1,
                            'left_time': original_index[lookback_idx],
                            # Use entire candle high as top
                            'top': df.loc[lookback_idx, 'high'],
                            # Use entire candle low as bottom
                            'bottom': df.loc[lookback_idx, 'low'],
                            'avg': (df.loc[lookback_idx, 'high'] + df.loc[lookback_idx, 'low']) / 2,
                            'height': df.loc[lookback_idx, 'high'] - df.loc[lookback_idx, 'low'],
                            'mitigated': False,
                            'mitigated_time': None,
                            'atr': df.loc[lookback_idx, 'atr'],
                            'height_atr_ratio': (df.loc[lookback_idx, 'high'] - df.loc[lookback_idx, 'low']) / df.loc[lookback_idx, 'atr'] if df.loc[lookback_idx, 'atr'] > 0 else 1,
                            'volume': df.loc[lookback_idx, 'volume'] if has_volume else 0,
                            'strength': calculate_ob_strength(df, lookback_idx, 1, historical_obs, has_volume)
                        }

                        # Add to list if strong enough
                        if ob['strength'] >= strength_threshold:
                            keep_ob, result = should_keep_ob(df, ob, len(
                                df)-1, use_should_keep_ob=use_should_keep_ob, analysis=analysis, strength_threshold=strength_threshold)
                            if keep_ob:
                                # Add score and quality info to the order block
                                ob['score'] = result["final_score"]
                                ob['setup_quality'] = result["setup_quality"]
                                ob['threshold'] = result["threshold"]
                                ob['warnings'] = result["warnings"]
                                ob['entry_quality'] = result.get(
                                    "entry_quality", "Unknown")
                                bullish_obs.append(ob)
                                historical_obs.append(ob)
                                break

        # Check for order block mitigation
        if idx > 0:
            # Set mitigation price depending on selected method
            if OBMitigationType == "Close":
                # Use close of previous bar
                bear_mitigation = df.loc[idx - 1, 'close']
                bull_mitigation = df.loc[idx - 1, 'close']
            else:  # "Wick"
                # Use current bar's high/low
                bear_mitigation = df.loc[idx, 'high']
                bull_mitigation = df.loc[idx, 'low']

            # Mitigate bearish order blocks
            mitigated_bearish = []
            for i, ob in enumerate(bearish_obs):
                # Bearish OB mitigated when price closes above the top of the OB
                if bear_mitigation > ob['top']:
                    ob['mitigated'] = True
                    ob['mitigated_time'] = original_index[idx if OBMitigationType ==
                                                          "Wick" else idx-1]
                    mitigated_bearish.append(i)
            bearish_obs = [ob for i, ob in enumerate(
                bearish_obs) if i not in mitigated_bearish]

            # Mitigate bullish order blocks
            mitigated_bullish = []
            for i, ob in enumerate(bullish_obs):
                # Bullish OB mitigated when price closes below the bottom of the OB
                if bull_mitigation < ob['bottom']:
                    ob['mitigated'] = True
                    ob['mitigated_time'] = original_index[idx if OBMitigationType ==
                                                          "Wick" else idx-1]
                    mitigated_bullish.append(i)
            bullish_obs = [ob for i, ob in enumerate(
                bullish_obs) if i not in mitigated_bullish]

        # Generate price alerts for active OBs
        if sell_alert:
            for ob in bearish_obs:
                if row['high'] > ob['bottom']:
                    print(
                        f"Sell alert at bar {idx}: Price entered bearish OB from bar {ob['index']}")

        if buy_alert:
            for ob in bullish_obs:
                if row['low'] < ob['top']:
                    print(
                        f"Buy alert at bar {idx}: Price entered bullish OB from bar {ob['index']}")

    # Limit number of order blocks and sort by strength
    if max_blocks > 0:
        # Allocate more slots to bearish OBs as per the image
        max_bearish = int(max_blocks * 0.6)
        max_bullish = max_blocks - max_bearish

        # Sort by strength and recency
        bearish_obs = sorted(
            bearish_obs, key=lambda x: (-x['strength'], x['left_time']), reverse=True)[:max_bearish]
        bullish_obs = sorted(
            bullish_obs, key=lambda x: (-x['strength'], x['left_time']), reverse=True)[:max_bullish]

    # Merge overlapping order blocks
    if merge_threshold > 0:
        bearish_obs = merge_overlapping_order_blocks(
            bearish_obs, merge_threshold)
        bullish_obs = merge_overlapping_order_blocks(
            bullish_obs, merge_threshold)

    # Final mitigation check - ensure no broken order blocks remain
    final_bearish_obs = []
    final_bullish_obs = []

    # Get the latest price data for final mitigation check
    latest_idx = len(df) - 1
    if latest_idx >= 0:
        latest_high = df.loc[latest_idx, 'high']
        latest_low = df.loc[latest_idx, 'low']
        latest_close = df.loc[latest_idx, 'close']

        # Final check for bearish order blocks
        for ob in bearish_obs:
            is_mitigated = False

            # Check if any price action after OB creation broke the top
            ob_idx = ob['index']
            for check_idx in range(ob_idx + 1, len(df)):
                if OBMitigationType == "Close":
                    mitigation_price = df.loc[check_idx, 'close']
                else:  # "Wick"
                    mitigation_price = df.loc[check_idx, 'high']

                if mitigation_price > ob['top']:
                    ob['mitigated'] = True
                    ob['mitigated_time'] = original_index[check_idx]
                    is_mitigated = True
                    break

            if not is_mitigated:
                final_bearish_obs.append(ob)

        # Final check for bullish order blocks
        for ob in bullish_obs:
            is_mitigated = False

            # Check if any price action after OB creation broke the bottom
            ob_idx = ob['index']
            for check_idx in range(ob_idx + 1, len(df)):
                if OBMitigationType == "Close":
                    mitigation_price = df.loc[check_idx, 'close']
                else:  # "Wick"
                    mitigation_price = df.loc[check_idx, 'low']

                if mitigation_price < ob['bottom']:
                    ob['mitigated'] = True
                    ob['mitigated_time'] = original_index[check_idx]
                    is_mitigated = True
                    break

            if not is_mitigated:
                final_bullish_obs.append(ob)
    else:
        final_bearish_obs = bearish_obs
        final_bullish_obs = bullish_obs

    # Return combined list of order blocks
    return final_bearish_obs + final_bullish_obs


def calculate_ob_strength(df, idx, direction, historical_obs, has_volume):
    """
    Calculate order block strength based on multiple factors.

    Parameters:
        df (pd.DataFrame): Price data
        idx (int): Index of the order block candle
        direction (int): Direction of the order block (1=bullish, -1=bearish)
        historical_obs (list): Previously detected order blocks
        has_volume (bool): Whether volume data is available

    Returns:
        int: Strength score (0-100)
    """
    # Calculate height
    height = df.at[idx, 'high'] - df.at[idx, 'low']
    atr = df.at[idx, 'atr']
    height_ratio = height / atr if atr > 0 else 1

    # 1. Calculate volume strength (40% of score)
    volume_strength = 0
    if has_volume:
        vol = df.at[idx, 'volume']
        vol_ma = df.at[idx, 'volume_ma']
        volume_ratio = vol / vol_ma if vol_ma > 0 else 1
        # Higher volume relative to average indicates stronger institutional activity
        volume_strength = min(volume_ratio * 40, 50)  # Cap at 50 instead of 60

    # 2. Calculate price range strength (30% of score)
    # Larger order blocks relative to ATR indicate stronger institutional presence
    price_range_strength = min(height_ratio * 30, 30)

    # 3. Calculate direction alignment strength (30% of score)
    direction_strength = 0
    if idx >= 10:  # Need enough history to analyze trend
        # Analyze recent candle directions (last 5-10 candles)
        lookback_period = min(10, idx)
        recent_bullish = 0
        recent_bearish = 0

        for i in range(1, lookback_period + 1):
            check_idx = idx - i
            if check_idx >= 0:
                candle_close = df.at[check_idx, 'close']
                candle_open = df.at[check_idx, 'open']

                if candle_close > candle_open:
                    recent_bullish += 1
                elif candle_close < candle_open:
                    recent_bearish += 1

        # Calculate recent trend bias
        total_candles = recent_bullish + recent_bearish
        if total_candles > 0:
            bullish_ratio = recent_bullish / total_candles
            bearish_ratio = recent_bearish / total_candles

            if direction == 1:  # Bullish OB
                if bullish_ratio > 0.6:
                    # OB aligns with trend - moderate strength
                    direction_strength = 20
                elif bearish_ratio > 0.6:
                    # Counter-trend OB - potentially strong reversal signal
                    direction_strength = 25
                else:
                    # Neutral/mixed trend - average strength
                    direction_strength = 15
            else:  # Bearish OB (direction == -1)
                if bearish_ratio > 0.6:
                    # OB aligns with trend - moderate strength
                    direction_strength = 20
                elif bullish_ratio > 0.6:
                    # Counter-trend OB - potentially strong reversal signal
                    direction_strength = 25
                else:
                    # Neutral/mixed trend - average strength
                    direction_strength = 15
        else:
            # No clear trend data - neutral strength
            direction_strength = 15

    # Calculate total strength
    total_strength = int(
        volume_strength + price_range_strength + direction_strength)

    # Ensure strength is within 0-100 range
    total_strength = max(0, min(100, total_strength))

    return total_strength


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

    # Plot bullish order blocks - green color as in the image
    for ob in all_bull_obs:
        start = ob['left_time']
        end = ob['mitigated_time'] or df.index[-1]
        plt.fill_betweenx([ob['bottom'], ob['top']],
                          start, end,
                          color='#64C4AC', alpha=0.15, edgecolor='#5db49e')
        plt.hlines(ob['avg'], start, end,
                   colors='#5db49e', linestyles='dashed', linewidth=1, alpha=0.5)

    # Plot bearish order blocks - blue color as in the image
    for ob in all_bear_obs:
        start = ob['left_time']
        end = ob['mitigated_time'] or df.index[-1]
        plt.fill_betweenx([ob['bottom'], ob['top']],
                          start, end,
                          color='#506CD3', alpha=0.15, edgecolor='#4760bb')
        plt.hlines(ob['avg'], start, end,
                   colors='#4760bb', linestyles='dashed', linewidth=1, alpha=0.5)

    # Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.grid(alpha=0.2)
    plt.title('Sonarlab Order Block Detection')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sonarlab Order Block Detector')
    parser.add_argument(
        '--symbol', type=str, default='SOLUSDT', help='Symbol to fetch data')
    parser.add_argument(
        '--interval', type=str, default='15m', help='Interval to fetch data')
    parser.add_argument(
        '--sensitivity', type=float, default=0.25, help='Sensitivity for order block detection (0.01-1.0)')
    parser.add_argument(
        '--mitigation', type=str, default='Wick', choices=['Wick', 'Close'],
        help='Method to determine when OBs are mitigated')
    parser.add_argument(
        '--days', type=int, default=10, help='Number of days of historical data')
    parser.add_argument(
        '--max_blocks', type=int, default=20, help='Maximum number of order blocks to display')
    parser.add_argument(
        '--use_should_keep_ob', type=str, default='True', help='Whether to use should_keep_ob for OB direction')

    args = parser.parse_args()
    client = Client()
    fetchData = BinanceDataFetcher(client)
    start_time = datetime.now() - timedelta(days=args.days)
    rawData = client.get_historical_klines(
        args.symbol, interval=args.interval, start_str=int(start_time.timestamp() * 1000), end_str=int((datetime.now()-timedelta(hours=5, minutes=30)).timestamp() * 1000))
    data = pd.DataFrame(rawData, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = data[col].astype(float)

    data.set_index('timestamp', inplace=True)

    # Verify the DataFrame has required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [
        col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}")

    # data = fetchData.get_historical_klines(
    #     args.symbol, interval=args.interval, start_time=start_time)

    # Detect order blocks with the improved algorithm
    order_blocks = detect_order_sensitive_blocks(
        data,
        sens=args.sensitivity,
        OBMitigationType=args.mitigation,
        max_blocks=args.max_blocks,
        merge_threshold=0.1,
        use_should_keep_ob=True if args.use_should_keep_ob == 'True' else False
    )

    # Separate order blocks by direction
    bullish_obs = [ob for ob in order_blocks if ob['direction'] == 1]
    bearish_obs = [ob for ob in order_blocks if ob['direction'] == -1]

    # print bullish and bearish start time
    for ob in bullish_obs:
        print(
            f"Bullish start time: {ob['left_time']}, top: {ob['top']}, bottom: {ob['bottom']}, avg: {ob['avg']}, height: {ob['height']}, volume: {ob['volume']}, strength: {ob['strength']}")
    for ob in bearish_obs:
        print(
            f"Bearish start time: {ob['left_time']}, top: {ob['top']}, bottom: {ob['bottom']}, avg: {ob['avg']}, height: {ob['height']}, volume: {ob['volume']}, strength: {ob['strength']}")

    plot_order_blocks(data, bullish_obs, bearish_obs)
