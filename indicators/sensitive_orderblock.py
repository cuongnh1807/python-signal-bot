import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from binance.client import Client

from binance_data_fetcher import BinanceDataFetcher
from indicators.candles import should_keep_ob
from indicators.rsi import calculate_macd
from helpers.price import merge_overlapping_order_blocks


def detect_order_sensitive_blocks(df, sens=0.28, OBMitigationType="Close", buy_alert=False, sell_alert=False, volume_lookback=20, merge_threshold=0.5, max_blocks=10, atr_period=14, strength_threshold=70):
    """
    Detect bullish and bearish order blocks in a financial dataset based on Pine Script logic.

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

    Returns:
    - list: Combined list of active order blocks (both bearish and bullish).
    """
    # Lưu trữ index gốc trước khi reset
    original_index = df.index.copy()

    # Đảm bảo DataFrame có index số nguyên cho tính toán
    df = df.reset_index(drop=True)
    macd_info = calculate_macd(df)
    df['macd'] = macd_info['macd']
    df['macd_signal'] = macd_info['signal']
    df['macd_hist'] = macd_info['histogram']

    # Tính toán khối lượng trung bình nếu có dữ liệu khối lượng
    has_volume = 'volume' in df.columns
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    # Tính toán ATR cho height_strength
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['atr'] = df['atr'].fillna(df['tr'].mean())

    # Calculate ROC
    df['pc'] = (df['open'] - df['open'].shift(4)) / df['open'].shift(4) * 100
    df['pc_close'] = (df['close'] - df['close'].shift(4)) / \
        df['close'].shift(4) * 100

    # Detect crossunders and crossovers
    df['crossunder_open'] = (df['pc'].shift(1) > -sens) & (df['pc'] <= -sens)
    df['crossunder_close'] = (df['pc_close'].shift(
        1) > -sens) & (df['pc_close'] <= -sens)
    df['crossunder'] = df['crossunder_open'] | df['crossunder_close']

    df['crossover_open'] = (df['pc'].shift(1) < sens) & (df['pc'] >= sens)
    df['crossover_close'] = (df['pc_close'].shift(
        1) < sens) & (df['pc_close'] >= sens)
    df['crossover'] = df['crossover_open'] | df['crossover_close']

    # Lưu lịch sử các order block cho việc tính toán historical_count
    historical_obs = []

    # Initialize lists for active order blocks
    bearish_obs = []
    bullish_obs = []

    # Biến để theo dõi vị trí của crossunder/crossover gần nhất
    last_cross_bearish = None
    last_cross_bullish = None

    # Process each bar
    for idx, row in df.iterrows():
        # Bearish order block creation
        if row['crossunder']:
            if last_cross_bearish is None or (idx - last_cross_bearish) > 3:
                last_cross_bearish = idx
                for i in range(4, 16):
                    lookback_idx = idx - i
                    if lookback_idx < 0:
                        break
                    if df.loc[lookback_idx, 'close'] > df.loc[lookback_idx, 'open']:
                        candle_size = abs(
                            df.loc[lookback_idx, 'close'] - df.loc[lookback_idx, 'open'])
                        avg_size = (df.loc[max(0, lookback_idx-10):lookback_idx, 'high'].mean() -
                                    df.loc[max(0, lookback_idx-10):lookback_idx, 'low'].mean())
                        if candle_size > 0.15 * avg_size:
                            ob = create_block(
                                df, lookback_idx, -1, original_index, has_volume, historical_obs)
                            if ob['strength'] >= strength_threshold and should_keep_ob(df, ob, len(df) - 1):
                                bearish_obs.append(ob)
                                historical_obs.append(ob)
                        break

        # Bullish order block creation
        if row['crossover']:
            if last_cross_bullish is None or (idx - last_cross_bullish) > 3:
                last_cross_bullish = idx
                for i in range(4, 16):
                    lookback_idx = idx - i
                    if lookback_idx < 0:
                        break
                    if df.loc[lookback_idx, 'close'] < df.loc[lookback_idx, 'open']:
                        candle_size = abs(
                            df.loc[lookback_idx, 'close'] - df.loc[lookback_idx, 'open'])
                        avg_size = (df.loc[max(0, lookback_idx-10):lookback_idx, 'high'].mean() -
                                    df.loc[max(0, lookback_idx-10):lookback_idx, 'low'].mean())
                        if candle_size > 0.15 * avg_size:
                            ob = create_block(
                                df, lookback_idx, 1, original_index, has_volume, historical_obs)

                            if ob['strength'] >= strength_threshold and should_keep_ob(df, ob, len(df) - 1):
                                bullish_obs.append(ob)
                                historical_obs.append(ob)
                        break

        if idx > 0:
            if OBMitigationType == "Close":
                bear_mitigation = df.loc[idx - 1, 'close']
                bull_mitigation = df.loc[idx - 1, 'close']
            else:  # "Wick"
                bear_mitigation = df.loc[idx, 'high']  # Current bar's high
                bull_mitigation = df.loc[idx, 'low']   # Current bar's low

            # Remove mitigated bearish order blocks
            mitigated_bearish = []
            for i, ob in enumerate(bearish_obs):
                if bear_mitigation > ob['top']:
                    ob['mitigated_time'] = original_index[idx -
                                                          1] if OBMitigationType == "Close" else original_index[idx]
                    mitigated_bearish.append(i)
            bearish_obs = [ob for i, ob in enumerate(
                bearish_obs) if i not in mitigated_bearish]

            # Remove mitigated bullish order blocks
            mitigated_bullish = []
            for i, ob in enumerate(bullish_obs):
                if bull_mitigation < ob['bottom']:
                    ob['mitigated_time'] = original_index[idx -
                                                          1] if OBMitigationType == "Close" else original_index[idx]
                    mitigated_bullish.append(i)
            bullish_obs = [ob for i, ob in enumerate(
                bullish_obs) if i not in mitigated_bullish]

        # Alerts for active order blocks
        for ob in bearish_obs:
            if row['high'] > ob['bottom'] and sell_alert:
                print(
                    f"Sell alert at bar {idx}: Price entered bearish OB from bar {ob['index']}")

        for ob in bullish_obs:
            if row['low'] < ob['top'] and buy_alert:
                print(
                    f"Buy alert at bar {idx}: Price entered bullish OB from bar {ob['index']}")

    if max_blocks > 0:
        max_bearish = int(max_blocks * 0.6)
        max_bullish = max_blocks - max_bearish
        bearish_obs = sorted(bearish_obs, key=lambda x: (
            -x['strength'] if x['strength'] else 0, x['left_time']), reverse=True)[:max_bearish]
        bullish_obs = sorted(bullish_obs, key=lambda x: (
            -x['strength'] if x['strength'] else 0, x['left_time']), reverse=True)[:max_bullish]

    # Gộp các order block chồng lấp
    if merge_threshold > 0:
        bearish_obs = merge_overlapping_order_blocks(
            bearish_obs, merge_threshold)
        bullish_obs = merge_overlapping_order_blocks(
            bullish_obs, merge_threshold)

    return bearish_obs + bullish_obs


def create_block(df, idx, direction, original_index, has_volume, historical_obs=None):
    # Tính toán kích thước thực của nến
    candle_body = abs(df.at[idx, 'close'] - df.at[idx, 'open'])
    candle_range = df.at[idx, 'high'] - df.at[idx, 'low']
    body_percent = candle_body / candle_range if candle_range > 0 else 0

    # Tính chiều cao của OB
    top = df.at[idx, 'high']
    bottom = df.at[idx, 'low']
    height = top - bottom

    ob = {
        'index': idx,
        'left_time': original_index[idx],
        'top': top,
        'bottom': bottom,
        'direction': direction,
        'mitigated_time': None,
        'avg': (top + bottom) / 2,
        'height': height,
        'atr': df['atr'].iloc[idx] or 0,
        'height_atr_ratio': (top - bottom) / df['atr'].iloc[idx],
        'body_size': candle_body,
        'volume': 0,
        'historical_count': 0,
        'recent_count': 0,
        'strength': 0
    }

    # 1. Volume strength
    volume_strength = 0
    if has_volume:
        vol = df.at[df.index[idx], 'volume']
        ob['volume'] = vol
        volume_ma = df.at[df.index[idx], 'volume_ma']
        volume_ratio = vol / volume_ma if volume_ma > 0 else 1
        volume_strength = min(volume_ratio * 40, 40)  # Tối đa 40 điểm

    # 2. Height strength
    height_ratio = ob['height_atr_ratio']
    height_strength = min(height_ratio * 35, 35)  # Tối đa 35 điểm

    # 3. Historical strength
    historical_count = 0
    recent_count = 0
    if historical_obs:
        ob_price_range = height * 1.5

        for other_ob in historical_obs:
            other_avg = (other_ob['top'] + other_ob['bottom']) / 2
            if abs(ob['avg'] - other_avg) <= ob_price_range:
                historical_count += 1
                # Xác định 'gần đây' dựa vào index
                if abs(idx - other_ob['index']) <= 100:
                    recent_count += 1

    ob['historical_count'] = historical_count
    ob['recent_count'] = recent_count

    if recent_count >= 3:
        historical_strength = max(5, 15 - (recent_count - 2) * 5)
    else:
        historical_strength = min(historical_count * 4, 25)

    ob['strength'] = int(
        volume_strength + height_strength + historical_strength)

    return ob


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-ticker trading bot')
    parser.add_argument(
        '--symbol', type=str, default='SOLUSDT', help='Symbol to fetch data')
    parser.add_argument(
        '--interval', type=str, default='15m', help='Interval to fetch data')
    args = parser.parse_args()
    client = Client()
    fetchData = BinanceDataFetcher(client)
    start_time = datetime.now() - timedelta(days=10)
    data = fetchData.get_historical_klines(
        args.symbol, interval=args.interval, start_time=start_time)
    order_blocks = detect_order_sensitive_blocks(
        data, 0.28, merge_threshold=0.7)

    # Tách order blocks theo hướng
    bullish_obs = [ob for ob in order_blocks if ob['direction'] == 1]
    bearish_obs = [ob for ob in order_blocks if ob['direction'] == -1]

    print("Active Bearish Order Blocks:", bearish_obs)
    print("Active Bullish Order Blocks:", bullish_obs)
    plot_order_blocks(data, bullish_obs, bearish_obs)
