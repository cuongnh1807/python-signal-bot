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
    volume_lookback=20
):
    """
    Detect Order Blocks in OHLCV data.

    Parameters:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        length (int): Number of bars for pivot detection.
        bull_ext_last (int): Number of recent bullish OBs to track.
        bear_ext_last (int): Number of recent bearish OBs to track.
        mitigation_method (str): 'Wick' or 'Close' price for mitigation.
        volume_lookback (int): Lookback period for volume moving average.

    Returns:
        pd.DataFrame: Updated DataFrame with OB signals and levels.
    """

    df = df.copy()

    # Kiểm tra xem có dữ liệu khối lượng không
    has_volume = 'volume' in df.columns

    # Tính toán khối lượng trung bình nếu có dữ liệu khối lượng
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        # Điền giá trị NaN bằng giá trị đầu tiên có sẵn
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    # Calculate indicators
    df['upper'] = df['high'].rolling(length).max()
    df['lower'] = df['low'].rolling(length).min()
    df['high_length_ago'] = df['high'].shift(length)
    df['low_length_ago'] = df['low'].shift(length)

    # Calculate OS (market structure)
    df['os'] = np.nan
    df.loc[df.index[length], 'os'] = 0  # Initial value

    for i in range(length+1, len(df)):
        high_la = df['high_length_ago'].iloc[i]
        low_la = df['low_length_ago'].iloc[i]
        upper = df['upper'].iloc[i]
        lower = df['lower'].iloc[i]

        if high_la > upper:
            df.at[df.index[i], 'os'] = 0
        elif low_la < lower:
            df.at[df.index[i], 'os'] = 1
        else:
            df.at[df.index[i], 'os'] = df['os'].iloc[i-1]

    # Detect pivot highs in volume
    df['phv'] = False
    for i in range(length, len(df) - length):
        window = df['volume'].iloc[i-length:i+length+1]
        if df['volume'].iloc[i] == window.max():
            df.at[df.index[i], 'phv'] = True

    # Order blocks storage
    bull_obs = []
    bear_obs = []

    # Signals columns
    df['bull_ob'] = np.nan
    df['bear_ob'] = np.nan
    df['mitigated_bull'] = False
    df['mitigated_bear'] = False

    for i in range(len(df)):
        current_time = df.index[i]

        # Detect new OBs
        if df['phv'].iloc[i]:
            os = df['os'].iloc[i]
            left_idx = i - length
            if left_idx >= 0:
                if os == 1:  # Bullish OB
                    top = (df['high'].iloc[left_idx] +
                           df['low'].iloc[left_idx]) / 2
                    bottom = df['low'].iloc[left_idx]
                    avg = (top + bottom) / 2

                    # Tạo order block với thông tin khối lượng
                    ob = {
                        'top': top,
                        'bottom': bottom,
                        'avg': avg,
                        'left_time': df.index[left_idx],
                        'mitigated_time': None,
                        'mitigated': False,
                        'direction': 'bullish',
                        'volume': 0,  # Khởi tạo giá trị mặc định
                        'strength': 0  # Khởi tạo giá trị mặc định
                    }

                    # Thêm thông tin khối lượng nếu có
                    if has_volume:
                        obVolume = df['volume'].iloc[left_idx]
                        if left_idx + 1 < len(df):
                            obVolume += df['volume'].iloc[left_idx + 1]
                        if left_idx + 2 < len(df):
                            obVolume += df['volume'].iloc[left_idx + 2]

                        ob['volume'] = obVolume

                        # Tính toán sức mạnh dựa trên khối lượng trung bình
                        if df['volume_ma'].iloc[left_idx] > 0:
                            percentage = min(
                                int((obVolume / df['volume_ma'].iloc[left_idx]) * 100), 100)
                            ob['strength'] = percentage

                    bull_obs.insert(0, ob)
                    df.at[current_time, 'bull_ob'] = bottom

                else:  # Bearish OB
                    bottom = (df['high'].iloc[left_idx] +
                              df['low'].iloc[left_idx]) / 2
                    top = df['high'].iloc[left_idx]
                    avg = (top + bottom) / 2

                    # Tạo order block với thông tin khối lượng
                    ob = {
                        'top': top,
                        'bottom': bottom,
                        'avg': avg,
                        'left_time': df.index[left_idx],
                        'mitigated_time': None,
                        'mitigated': False,
                        'direction': 'bearish',
                        'volume': 0,  # Khởi tạo giá trị mặc định
                        'strength': 0  # Khởi tạo giá trị mặc định
                    }

                    # Thêm thông tin khối lượng nếu có
                    if has_volume:
                        obVolume = df['volume'].iloc[left_idx]
                        if left_idx + 1 < len(df):
                            obVolume += df['volume'].iloc[left_idx + 1]
                        if left_idx + 2 < len(df):
                            obVolume += df['volume'].iloc[left_idx + 2]

                        ob['volume'] = obVolume

                        # Tính toán sức mạnh dựa trên khối lượng trung bình
                        if df['volume_ma'].iloc[left_idx] > 0:
                            percentage = min(
                                int((obVolume / df['volume_ma'].iloc[left_idx]) * 100), 100)
                            ob['strength'] = percentage

                    bear_obs.insert(0, ob)
                    df.at[current_time, 'bear_ob'] = top

        # Mitigation checks
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]

        # Check bullish OBs
        for ob in bull_obs[:]:
            if ((mitigation_method == 'Wick' and current_low < ob['bottom']) or
                    (mitigation_method == 'Close' and current_close < ob['bottom'])):
                ob['mitigated'] = True
                bull_obs.remove(ob)
                df.at[current_time, 'mitigated_bull'] = True

        # Check bearish OBs
        for ob in bear_obs[:]:
            if ((mitigation_method == 'Wick' and current_high > ob['top']) or
                    (mitigation_method == 'Close' and current_close > ob['top'])):
                ob['mitigated'] = True
                bear_obs.remove(ob)
                df.at[current_time, 'mitigated_bear'] = True

        # Keep only last X OBs
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
    fetchData = BinanceDataFetcher()
    start_time = datetime.now() - timedelta(days=7)
    data = fetchData.get_historical_klines(
        'BTCUSDT', interval="15m", start_time=start_time)
    # df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close',
    #                                  'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # df.set_index('timestamp', inplace=True)
    df, orders = detect_pivot_volume_order_blocks(data, length=5,
                                                  bull_ext_last=5, bear_ext_last=10)
    # Tách order blocks theo hướng
    bullish_obs = [ob for ob in orders if ob['direction'] == 'bullish']
    bearish_obs = [ob for ob in orders if ob['direction'] == 'bearish']
    print("Bullish Order Blocks:", bullish_obs)
    print("Bearish Order Blocks:", bearish_obs)
    plot_order_blocks(df, bullish_obs, bearish_obs)
