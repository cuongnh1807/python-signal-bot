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

from indicators.rsi import calculate_macd, calculate_rsi


def is_pin_bar(row):
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - max(row['open'], row['close'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    if body < 0.1 * (row['high'] - row['low']):
        if upper_shadow > 2 * body or lower_shadow > 2 * body:
            return True
    return False


def is_engulfing(df, i):
    if i > 0:
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        if (current['close'] > current['open'] and prev['close'] < prev['open'] and
                current['close'] > prev['open'] and current['open'] < prev['close']):
            return 'bullish'
        elif (current['close'] < current['open'] and prev['close'] > prev['open'] and
                current['close'] < prev['open'] and current['open'] > prev['close']):
            return 'bearish'
    return None


def should_keep_ob(df, ob, current_index):
    ob_direction = ob['direction']
    pin_bar_signal = is_pin_bar(df.iloc[current_index])
    engulfing_signal = is_engulfing(df, current_index)
    price_action_signal = pin_bar_signal or (engulfing_signal is not None)

    # 4. MACD Signal
    macd = df['macd'].iloc[current_index]
    macd_hist = df['macd_hist'].iloc[current_index]
    macd_signal_line = df['macd_signal'].iloc[current_index]

    if ob_direction == 'bullish':
        macd_signal = (macd < 0 and macd_hist > 0) or (
            macd > macd_signal_line and macd < 0)
    else:  # bearish
        macd_signal = (macd > 0 and macd_hist < 0) or (
            macd < macd_signal_line and macd > 0)
    signals = [price_action_signal, macd_signal]
    return sum(signals) >= 1


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
    strength_threshold=70,
):
    df = df.copy()

    has_volume = 'volume' in df.columns

    df['rsi'] = calculate_rsi(df)

    macd_info = calculate_macd(df)
    df['macd'] = macd_info['macd']
    df['macd_signal'] = macd_info['signal']
    df['macd_hist'] = macd_info['histogram']

    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['atr'] = df['atr'].fillna(df['tr'].mean())

    # Tính rolling highs và lows để phát hiện pivot
    df['upper'] = df['high'].rolling(length).max()
    df['lower'] = df['low'].rolling(length).min()

    if use_market_structure:
        df['os'] = 0
        for i in range(length, len(df)):
            high_prev = df['high'].iloc[i - length]
            low_prev = df['low'].iloc[i - length]
            upper = df['upper'].iloc[i]
            lower = df['lower'].iloc[i]
            if high_prev > upper:
                df.loc[df.index[i], 'os'] = 0
            elif low_prev < lower:
                df.loc[df.index[i], 'os'] = 1
            else:
                df.loc[df.index[i], 'os'] = df['os'].iloc[i - 1]

    df['phv'] = False
    if has_volume:
        for k in range(length, len(df) - length):
            if (df['volume'].iloc[k] > df['volume'].iloc[k - length:k].max() and
                    df['volume'].iloc[k] > df['volume'].iloc[k + 1:k + length + 1].max()):
                df.at[df.index[k], 'phv'] = True

    # Xác định mitigation targets dựa trên phương pháp
    if mitigation_method == 'Close':
        df['target_bull'] = df['close'].rolling(length).min()
        df['target_bear'] = df['close'].rolling(length).max()
    else:  # 'Wick'
        df['target_bull'] = df['low'].rolling(length).min()
        df['target_bear'] = df['high'].rolling(length).max()

    # Khởi tạo các cột cho OB signals và mitigation flags
    df['bull_ob'] = np.nan
    df['bear_ob'] = np.nan
    df['mitigated_bull'] = False
    df['mitigated_bear'] = False

    bull_obs = []
    bear_obs = []

    for i in range(2 * length, len(df)):
        current_time = df.index[i]

        if has_volume and df['phv'].iloc[i - length]:
            k = i - length
            direction = 'bullish' if df['os'].iloc[i] == 1 else 'bearish' if use_market_structure else (
                'bullish' if df['close'].iloc[i] > df['close'].iloc[i -
                                                                    length] else 'bearish'
            )

            if direction == 'bullish':
                top = (df['high'].iloc[k] + df['low'].iloc[k]) / 2
                bottom = df['low'].iloc[k]
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    top = bottom + min_height

                ob = {
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

                if has_volume:
                    valid_indices = [i for i in [k, k + 1] if i < len(df)]
                    vol = sum(df.at[df.index[i], 'volume'] for i in valid_indices) / \
                        len(valid_indices) if valid_indices else 0
                    ob['volume'] = vol
                    volume_k = df['volume'].iloc[k]
                    volume_ma = df['volume_ma'].iloc[len(df) - 1]
                    volume_ratio = volume_k / volume_ma if volume_ma > 0 else 1
                    height_ratio = ob['height'] / \
                        ob['atr'] if ob['atr'] > 0 else 1

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

                    if recent_count >= 3:
                        historical_strength = max(
                            5, 15 - (recent_count - 2) * 5)
                    else:
                        historical_strength = min(ob_count * 4, 25)

                    volume_strength = min(volume_ratio * 40, 40)
                    height_strength = min(height_ratio * 35, 35)
                    ob['strength'] = int(
                        volume_strength + height_strength + historical_strength)

                    if ob['strength'] >= strength_threshold:
                        bull_obs.insert(0, ob)
                        df.at[current_time, 'bull_ob'] = bottom

            else:  # Bearish OB
                top = df['high'].iloc[k]
                bottom = (df['high'].iloc[k] + df['low'].iloc[k]) / 2
                height = top - bottom
                min_height = df['atr'].iloc[k] * min_height_multiplier
                if height < min_height:
                    bottom = top - min_height

                ob = {
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

                if has_volume:
                    valid_indices = [i for i in [k, k + 1] if i < len(df)]
                    vol = sum(df.at[df.index[i], 'volume'] for i in valid_indices) / \
                        len(valid_indices) if valid_indices else 0
                    ob['volume'] = vol
                    volume_k = df['volume'].iloc[k]
                    volume_ma = df['volume_ma'].iloc[len(df) - 1]
                    volume_ratio = volume_k / volume_ma if volume_ma > 0 else 1
                    height_ratio = ob['height'] / \
                        ob['atr'] if ob['atr'] > 0 else 1

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

                    if recent_count >= 3:
                        historical_strength = max(
                            5, 15 - (recent_count - 2) * 5)
                    else:
                        historical_strength = min(ob_count * 4, 20)

                    volume_strength = min(volume_ratio * 40, 40)
                    height_strength = min(height_ratio * 40, 40)
                    ob['strength'] = int(
                        volume_strength + height_strength + historical_strength)
                    print("ob['strength']", ob['strength'])

                    if ob['strength'] >= strength_threshold:
                        bear_obs.insert(0, ob)
                        df.at[current_time, 'bear_ob'] = top

        target_bull = df['target_bull'].iloc[i]
        target_bear = df['target_bear'].iloc[i]

        for ob in bull_obs[:]:
            if target_bull < ob['bottom']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bull_obs.remove(ob)
                df.at[current_time, 'mitigated_bull'] = True

        for ob in bear_obs[:]:
            if target_bear > ob['top']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bear_obs.remove(ob)
                df.at[current_time, 'mitigated_bear'] = True

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
    bullish_obs = [ob for ob in orders if ob['direction'] == 1]
    bearish_obs = [ob for ob in orders if ob['direction'] == -1]
    print("Bullish Order Blocks:", bullish_obs)
    print("Bearish Order Blocks:", bearish_obs)
    plot_order_blocks(df, bullish_obs, bearish_obs)
