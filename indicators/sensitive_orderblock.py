from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from binance_data_fetcher import BinanceDataFetcher


def merge_overlapping_order_blocks(order_blocks, threshold=0.7):
    """
    Gộp các order block chồng lấp dựa trên mức độ chồng lấp về giá.

    Parameters:
        order_blocks (list): Danh sách các order block
        threshold (float): Ngưỡng chồng lấp để gộp (0-1), mặc định là 0.5 (50%)

    Returns:
        list: Danh sách các order block sau khi gộp
    """
    if not order_blocks:
        return []

    # Sắp xếp order blocks theo thời gian bắt đầu
    sorted_obs = sorted(order_blocks, key=lambda x: x['left_time'])
    merged_obs = []

    i = 0
    while i < len(sorted_obs):
        current_ob = sorted_obs[i]

        # Kiểm tra xem có thể gộp với order block tiếp theo không
        j = i + 1
        while j < len(sorted_obs):
            next_ob = sorted_obs[j]

            # Tính toán mức độ chồng lấp
            current_range = current_ob['top'] - current_ob['bottom']
            next_range = next_ob['top'] - next_ob['bottom']

            # Tìm phần chồng lấp
            overlap_top = min(current_ob['top'], next_ob['top'])
            overlap_bottom = max(current_ob['bottom'], next_ob['bottom'])

            if overlap_bottom < overlap_top:  # Có chồng lấp
                overlap_range = overlap_top - overlap_bottom
                overlap_ratio = overlap_range / min(current_range, next_range)

                # Nếu chồng lấp đủ lớn, gộp chúng lại
                if overlap_ratio >= threshold:
                    # Tạo order block mới từ việc gộp
                    current_ob = {
                        'index': min(current_ob['index'], next_ob['index']),
                        'top': max(current_ob['top'], next_ob['top']),
                        'bottom': min(current_ob['bottom'], next_ob['bottom']),
                        'left_time': min(current_ob['left_time'], next_ob['left_time']),
                        'direction': current_ob['direction'],
                        'mitigated_time': None,
                        'avg': (max(current_ob['top'], next_ob['top']) +
                                min(current_ob['bottom'], next_ob['bottom'])) / 2,
                        'volume': max(current_ob['volume'] or 0, next_ob['volume'] or 0),
                        'strength': max(current_ob['strength'] or 0, next_ob['strength'] or 0)
                    }

                    # Xóa order block đã gộp và tiếp tục kiểm tra
                    sorted_obs.pop(j)
                else:
                    j += 1
            else:
                j += 1

        merged_obs.append(current_ob)
        i += 1

    return merged_obs


def detect_order_sensitive_blocks(df, sens=0.28, OBMitigationType="Close", buy_alert=True, sell_alert=True, volume_lookback=20, merge_threshold=0.5, max_blocks=10):
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

    Returns:
    - list: Combined list of active order blocks (both bearish and bullish).
    """
    # Lưu trữ index gốc trước khi reset
    original_index = df.index.copy()

    # Đảm bảo DataFrame có index số nguyên cho tính toán
    df = df.reset_index(drop=True)

    # Tính toán khối lượng trung bình nếu có dữ liệu khối lượng
    has_volume = 'volume' in df.columns
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(volume_lookback).mean()
        # Điền giá trị NaN bằng giá trị đầu tiên có sẵn
        df['volume_ma'] = df['volume_ma'].fillna(
            df['volume'].iloc[0] if len(df) > 0 else 0)

    # Calculate ROC - tương tự PineScript: pc = (open - open[4]) / open[4] * 100
    df['pc'] = (df['open'] - df['open'].shift(4)) / df['open'].shift(4) * 100

    # Thêm ROC dựa trên giá đóng cửa để phát hiện tốt hơn
    df['pc_close'] = (df['close'] - df['close'].shift(4)) / \
        df['close'].shift(4) * 100

    # Detect crossunders and crossovers - tương tự PineScript ta.crossunder và ta.crossover
    # Sử dụng cả open và close để có nhiều tín hiệu hơn
    df['crossunder_open'] = (df['pc'].shift(1) > -sens) & (df['pc'] <= -sens)
    df['crossunder_close'] = (df['pc_close'].shift(
        1) > -sens) & (df['pc_close'] <= -sens)
    df['crossunder'] = df['crossunder_open'] | df['crossunder_close']

    df['crossover_open'] = (df['pc'].shift(1) < sens) & (df['pc'] >= sens)
    df['crossover_close'] = (df['pc_close'].shift(
        1) < sens) & (df['pc_close'] >= sens)
    df['crossover'] = df['crossover_open'] | df['crossover_close']

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
            # Giảm từ 5 xuống 3 để phát hiện nhiều OB hơn
            if last_cross_bearish is None or (idx - last_cross_bearish) > 3:
                last_cross_bearish = idx
                # Look back 4 to 15 bars for the first green candle (tương tự logic PineScript)
                for i in range(4, 16):
                    lookback_idx = idx - i
                    if lookback_idx < 0:
                        break
                    # Tìm nến xanh đầu tiên (bullish candle)
                    if df.loc[lookback_idx, 'close'] > df.loc[lookback_idx, 'open']:
                        # Giảm điều kiện kích thước để phát hiện nhiều OB hơn
                        candle_size = abs(
                            df.loc[lookback_idx, 'close'] - df.loc[lookback_idx, 'open'])
                        avg_size = df.loc[max(0, lookback_idx-10):lookback_idx, 'high'].mean(
                        ) - df.loc[max(0, lookback_idx-10):lookback_idx, 'low'].mean()

                        if candle_size > 0.15 * avg_size:  # Giảm xuống 0.15 thay vì 0.3
                            ob = create_block(
                                df, lookback_idx, 'bearish', original_index, has_volume)
                            bearish_obs.append(ob)
                        break

        # Bullish order block creation
        if row['crossover']:
            # Giảm từ 5 xuống 3
            if last_cross_bullish is None or (idx - last_cross_bullish) > 3:
                last_cross_bullish = idx
                # Look back 4 to 15 bars for the first red candle
                for i in range(4, 16):
                    lookback_idx = idx - i
                    if lookback_idx < 0:
                        break
                    # Tìm nến đỏ đầu tiên (bearish candle)
                    if df.loc[lookback_idx, 'close'] < df.loc[lookback_idx, 'open']:
                        # Giảm điều kiện kích thước
                        candle_size = abs(
                            df.loc[lookback_idx, 'close'] - df.loc[lookback_idx, 'open'])
                        avg_size = df.loc[max(0, lookback_idx-10):lookback_idx, 'high'].mean(
                        ) - df.loc[max(0, lookback_idx-10):lookback_idx, 'low'].mean()

                        if candle_size > 0.15 * avg_size:  # Giảm xuống 0.15
                            ob = create_block(
                                df, lookback_idx, 'bullish', original_index, has_volume)
                            bullish_obs.append(ob)
                        break

        # Mitigation check (skip first bar due to shift)
        if idx > 0:
            # Xác định giá trị sử dụng cho việc vô hiệu hóa order block dựa trên OBMitigationType
            if OBMitigationType == "Close":
                bear_mitigation = df.loc[idx - 1, 'close']
                bull_mitigation = df.loc[idx - 1, 'close']
            else:  # "Wick"
                bear_mitigation = df.loc[idx - 1, 'high']
                bull_mitigation = df.loc[idx - 1, 'low']

            # Remove mitigated bearish order blocks
            mitigated_bearish = []
            for i, ob in enumerate(bearish_obs):
                if bear_mitigation > ob['top']:
                    # Lưu thời gian vô hiệu hóa
                    ob['mitigated_time'] = original_index[idx - 1]
                    mitigated_bearish.append(i)

            # Xóa các order block đã bị vô hiệu hóa từ danh sách active
            bearish_obs = [ob for i, ob in enumerate(
                bearish_obs) if i not in mitigated_bearish]

            # Remove mitigated bullish order blocks
            mitigated_bullish = []
            for i, ob in enumerate(bullish_obs):
                if bull_mitigation < ob['bottom']:
                    # Lưu thời gian vô hiệu hóa
                    ob['mitigated_time'] = original_index[idx - 1]
                    mitigated_bullish.append(i)

            # Xóa các order block đã bị vô hiệu hóa từ danh sách active
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

    # Giới hạn số lượng order block nhưng ưu tiên phát hiện bearish OB nhiều hơn
    if max_blocks > 0:
        # Phân bổ lại tỷ lệ: 60% cho bearish, 40% cho bullish
        max_bearish = int(max_blocks * 0.6)
        max_bullish = max_blocks - max_bearish

        # Sắp xếp theo độ quan trọng (thời gian gần đây và/hoặc độ mạnh)
        bearish_obs = sorted(bearish_obs, key=lambda x: (
            -x['strength'] if x['strength'] else 0, x['left_time']), reverse=True)[:max_bearish]
        bullish_obs = sorted(bullish_obs, key=lambda x: (
            -x['strength'] if x['strength'] else 0, x['left_time']), reverse=True)[:max_bullish]

    # Sau khi xử lý tất cả các thanh, gộp các order block chồng lấp
    if merge_threshold > 0:
        bearish_obs = merge_overlapping_order_blocks(
            bearish_obs, merge_threshold)
        bullish_obs = merge_overlapping_order_blocks(
            bullish_obs, merge_threshold)

    return bearish_obs + bullish_obs


def create_block(df, idx, direction, original_index, has_volume):
    """Tạo order block dictionary với đầy đủ thông tin"""
    # Tính toán kích thước thực của nến
    candle_body = abs(df.at[idx, 'close'] - df.at[idx, 'open'])
    candle_range = df.at[idx, 'high'] - df.at[idx, 'low']
    body_percent = candle_body / candle_range if candle_range > 0 else 0

    # Tính momentum của nến
    momentum = 0
    if idx > 0:
        prev_range = df.at[idx-1, 'high'] - df.at[idx-1, 'low']
        momentum_ratio = candle_range / prev_range if prev_range > 0 else 1
        momentum = min(int(momentum_ratio * 100), 200)

    ob = {
        'index': idx,
        'left_time': original_index[idx],
        'top': df.at[idx, 'high'],
        'bottom': df.at[idx, 'low'],
        'direction': direction,
        'mitigated_time': None,
        'volume': 0,
        # Kết hợp body% và momentum
        'strength': int((body_percent * 100 + momentum) / 2),
        'avg': (df.at[idx, 'high'] + df.at[idx, 'low'])/2,
        'body_size': candle_body
    }

    if has_volume:
        vol = sum(df.at[i, 'volume']
                  for i in [idx, idx+1, idx+2] if i < len(df))
        ob['volume'] = vol
        vol_ma = df.at[idx, 'volume_ma']
        if vol_ma > 0:
            # Kết hợp độ mạnh từ khối lượng và kích thước nến
            vol_strength = min(int((vol / vol_ma) * 100), 100)
            ob['strength'] = int((vol_strength + ob['strength']) / 2)

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
    fetchData = BinanceDataFetcher()
    start_time = datetime.now() - timedelta(days=10)
    data = fetchData.get_historical_klines(
        "BTCUSDT", interval="15m", start_time=start_time)
    order_blocks = detect_order_sensitive_blocks(
        data, 0.4, merge_threshold=0.7)

    # Tách order blocks theo hướng
    bullish_obs = [ob for ob in order_blocks if ob['direction'] == 'bullish']
    bearish_obs = [ob for ob in order_blocks if ob['direction'] == 'bearish']

    print("Active Bearish Order Blocks:", bearish_obs)
    print("Active Bullish Order Blocks:", bullish_obs)
    plot_order_blocks(data, bullish_obs, bearish_obs)
