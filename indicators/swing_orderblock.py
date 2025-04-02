import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from helpers.price import merge_overlapping_order_blocks
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from binance.client import Client
from binance_data_fetcher import BinanceDataFetcher
from indicators.rsi import calculate_macd, calculate_rsi


def create_order_block(
    index: int,
    direction: int,
    left_time: datetime,
    top: float,
    bottom: float,
    atr: float = 0,
    volume: float = 0,
    strength: float = 0
) -> Dict[str, Any]:
    """Create an order block dictionary with all necessary properties"""
    return {
        'index': index,
        'direction': direction,
        'left_time': left_time,
        'top': top,
        'bottom': bottom,
        'avg': (top + bottom) / 2,
        'height': top - bottom,
        'mitigated': False,
        'mitigated_time': None,
        'breaker': False,
        'breaker_time': None,
        'strength': strength,
        'atr': atr,
        'volume': volume
    }


def detect_swing_points(df: pd.DataFrame, length: int = 10, use_body: bool = False) -> pd.DataFrame:
    """
    Detect swing high and low points using the specified lookback period.

    Args:
        df: DataFrame with OHLCV data
        length: Lookback period for swing detection
        use_body: Whether to use candle body instead of wicks
    """
    df = df.copy()

    # Use candle body or wicks based on settings
    if use_body:
        df['max_price'] = df[['open', 'close']].max(axis=1)
        df['min_price'] = df[['open', 'close']].min(axis=1)
    else:
        df['max_price'] = df['high']
        df['min_price'] = df['low']

    # Calculate rolling highs and lows
    df['upper'] = df['high'].rolling(length).max()
    df['lower'] = df['low'].rolling(length).min()

    # Initialize market structure
    df['os'] = 0  # Market structure (0 for bearish, 1 for bullish)

    # Detect market structure changes
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
            df.loc[df.index[i], 'os'] = df['os'].iloc[i - 1]

    return df


def detect_swing_orderblocks(
    df: pd.DataFrame,
    length: int = 10,
    bull_ext_last: int = 3,
    bear_ext_last: int = 3,
    use_body: bool = False,
    strength_threshold: float = 70,
    volume_lookback: int = 20,
    atr_period: int = 14
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Detect order blocks based on swing points and market structure.

    Args:
        df: DataFrame with OHLCV data
        length: Lookback period for swing detection
        bull_ext_last: Number of bullish OBs to maintain
        bear_ext_last: Number of bearish OBs to maintain
        use_body: Whether to use candle body instead of wicks
        strength_threshold: Minimum strength score for valid OBs
        volume_lookback: Lookback period for volume MA
        atr_period: Period for ATR calculation
    """
    df = df.copy()

    # Ensure timestamp is properly set
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

    # Calculate technical indicators
    df['rsi'] = calculate_rsi(df)
    macd_info = calculate_macd(df)
    df['macd'] = macd_info['macd']
    df['macd_signal'] = macd_info['signal']
    df['macd_hist'] = macd_info['histogram']

    # Calculate volume MA
    if 'volume' in df.columns:
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

    # Detect swing points
    df = detect_swing_points(df, length, use_body)

    bull_obs: List[Dict[str, Any]] = []
    bear_obs: List[Dict[str, Any]] = []

    # Detect order blocks
    for i in range(2 * length, len(df)):
        current_time = df.index[i]

        # Check for structure change
        if df['os'].iloc[i] != df['os'].iloc[i-1]:
            k = i - 1  # Previous candle forms the order block

            if df['os'].iloc[i] == 1:  # Bullish structure change
                # Form bullish order block
                bottom = df['low'].iloc[k]
                top = (df['high'].iloc[k] + df['low'].iloc[k]) / 2

                ob = create_order_block(
                    index=k,
                    direction=1,
                    left_time=df.index[k],
                    top=top,
                    bottom=bottom,
                    atr=df['atr'].iloc[k],
                    volume=df['volume'].iloc[k] if 'volume' in df.columns else 0,
                    strength=calculate_ob_strength(df, k, top, bottom, True)
                )

                if ob['strength'] >= strength_threshold:
                    bull_obs.insert(0, ob)

            else:  # Bearish structure change
                # Form bearish order block
                top = df['high'].iloc[k]
                bottom = (df['high'].iloc[k] + df['low'].iloc[k]) / 2

                ob = create_order_block(
                    index=k,
                    direction=-1,
                    left_time=df.index[k],
                    top=top,
                    bottom=bottom,
                    atr=df['atr'].iloc[k],
                    volume=df['volume'].iloc[k] if 'volume' in df.columns else 0,
                    strength=calculate_ob_strength(df, k, top, bottom, False)
                )

                if ob['strength'] >= strength_threshold:
                    bear_obs.insert(0, ob)

        # Check for breaker blocks and mitigation
        current_price = df['close'].iloc[i]

        # Update bullish OBs
        for ob in bull_obs[:]:
            if not ob['breaker'] and current_price < ob['bottom']:
                ob['breaker'] = True
                ob['breaker_time'] = current_time
            elif ob['breaker'] and current_price > ob['top']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bull_obs.remove(ob)

        # Update bearish OBs
        for ob in bear_obs[:]:
            if not ob['breaker'] and current_price > ob['top']:
                ob['breaker'] = True
                ob['breaker_time'] = current_time
            elif ob['breaker'] and current_price < ob['bottom']:
                ob['mitigated'] = True
                ob['mitigated_time'] = current_time
                bear_obs.remove(ob)

        # Maintain limited number of OBs
        bull_obs = bull_obs[:bull_ext_last]
        bear_obs = bear_obs[:bear_ext_last]

        # Merge overlapping blocks
        bull_obs = merge_overlapping_order_blocks(bull_obs, 0.5)
        bear_obs = merge_overlapping_order_blocks(bear_obs, 0.5)

    return df, bull_obs + bear_obs


def calculate_ob_strength(
    df: pd.DataFrame,
    index: int,
    top: float,
    bottom: float,
    is_bullish: bool
) -> float:
    """
    Calculate strength score for an order block based on various factors.

    Args:
        df: DataFrame with OHLCV data
        index: Bar index of the OB
        top: Top price of OB
        bottom: Bottom price of OB
        is_bullish: Whether OB is bullish
    """
    height = top - bottom
    avg_price = (top + bottom) / 2

    # Volume factor (if volume data available)
    volume_score = 0
    if 'volume' in df.columns:
        volume_ratio = df['volume'].iloc[index] / \
            df['volume'].iloc[index-20:index].mean()
        volume_score = min(volume_ratio * 40, 40)

    # Height factor
    if 'atr' in df.columns:
        height_ratio = height / df['atr'].iloc[index]
        height_score = min(height_ratio * 35, 35)
    else:
        avg_height = df['high'].iloc[index-20:index].mean() - \
            df['low'].iloc[index-20:index].mean()
        height_ratio = height / avg_height
        height_score = min(height_ratio * 35, 35)

    # Historical significance
    price_range = height * 1.5
    historical_count = 0
    recent_count = 0

    for j in range(max(0, index - 500), index):
        if j >= 0:
            candle_avg = (df['high'].iloc[j] + df['low'].iloc[j]) / 2
            if abs(avg_price - candle_avg) <= price_range:
                historical_count += 1
                if j >= max(0, index - 100):
                    recent_count += 1

    if recent_count >= 3:
        historical_score = max(5, 15 - (recent_count - 2) * 5)
    else:
        historical_score = min(historical_count * 4, 25)

    return volume_score + height_score + historical_score


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
    fetchData = BinanceDataFetcher(client=client)
    start_time = datetime.now() - timedelta(days=7)
    data = fetchData.get_historical_klines(
        args.symbol, interval=args.interval, start_time=start_time)
    # df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close',
    #                                  'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # df.set_index('timestamp', inplace=True)
    df, orders = detect_swing_orderblocks(data, length=10,
                                          bull_ext_last=5, bear_ext_last=10,
                                          use_body=True)
    # Tách order blocks theo hướng
    bullish_obs = [ob for ob in orders if ob['direction'] == 1]
    bearish_obs = [ob for ob in orders if ob['direction'] == -1]
    print("Bullish Order Blocks:", bullish_obs)
    print("Bearish Order Blocks:", bearish_obs)
    plot_order_blocks(df, bullish_obs, bearish_obs)
