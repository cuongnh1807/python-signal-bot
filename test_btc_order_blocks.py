from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from binance_data_fetcher import BinanceDataFetcher
import pandas as pd
# --- Inputs (mirroring Pine Script settings) ---
length = 5                # Volume Pivot Length
bull_ext_last = 5        # Number of last bullish order blocks to extend
bear_ext_last = 3         # Number of last bearish order blocks to extend
mitigation = 'Wick'       # Mitigation method: 'Wick' or 'Close'
bg_bull_css = '#169400'   # Background color for bullish order blocks
bull_css = '#169400'      # Border color for bullish order blocks
bull_avg_css = '#9598a1'  # Average line color for bullish order blocks
bg_bear_css = '#ff1100'   # Background color for bearish order blocks
bear_css = '#ff1100'      # Border color for bearish order blocks
bear_avg_css = '#9598a1'  # Average line color for bearish order blocks

# --- Step 1: Fetch BTCUSDT Data (using BTC-USD from Yahoo Finance) ---
symbol = 'BTCUSDT'
fetchData = BinanceDataFetcher()
start_time = datetime.now() - timedelta(days=9)
data = fetchData.get_historical_klines(
    symbol, interval="15m", start_time=start_time)

# --- Step 2: Calculate Indicators ---
# Highest high and lowest low over 'length' bars
# --- Calculate Indicators ---
data['upper'] = data['high'].rolling(window=length, min_periods=1).max()
data['lower'] = data['low'].rolling(window=length, min_periods=1).min()

# Oscillator state 'os'
os = [0] * len(data)
for i in range(1, len(data)):
    high_shifted = data['high'].shift(length).iloc[i]
    low_shifted = data['low'].shift(length).iloc[i]
    upper_current = data['upper'].iloc[i]
    lower_current = data['lower'].iloc[i]
    if pd.notna(high_shifted) and high_shifted > upper_current:
        os[i] = 0
    elif pd.notna(low_shifted) and low_shifted < lower_current:
        os[i] = 1
    else:
        os[i] = os[i - 1]
data['os'] = os

# Volume pivot high 'phv'
phv = [False] * len(data)
phv_count = 0
for i in range(length, len(data) - length):
    vol_current = data['volume'].iloc[i]
    vol_left = data['volume'].iloc[i - length:i].max()
    vol_right = data['volume'].iloc[i + 1:i + length + 1].max()
    if vol_current > vol_left and vol_current > vol_right:
        phv[i] = True
        phv_count += 1
data['phv'] = phv
print(f"Total Volume Pivot Highs Detected: {phv_count}")

# Mitigation targets
if mitigation == 'Close':
    data['target_bull'] = data['close']
    data['target_bear'] = data['close']
else:  # 'Wick'
    data['target_bull'] = data['low']
    data['target_bear'] = data['high']

# --- Dynamic Order Block Detection and Mitigation ---
bull_order_blocks = []
bear_order_blocks = []

for i in range(length, len(data)):
    # Add new order blocks if detected
    if data['phv'].iloc[i]:
        if data['os'].iloc[i] == 1:  # Bullish order block
            top = (data['high'].iloc[i - length] +
                   data['low'].iloc[i - length]) / 2
            btm = data['low'].iloc[i - length]
            avg = (top + btm) / 2
            left = data.index[i - length]
            bull_order_blocks.append({
                'top': top,
                'btm': btm,
                'avg': avg,
                'left': left,
                'detection_index': i
            })
        elif data['os'].iloc[i] == 0:  # Bearish order block
            top = data['high'].iloc[i - length]
            btm = (data['high'].iloc[i - length] +
                   data['low'].iloc[i - length]) / 2
            avg = (top + btm) / 2
            left = data.index[i - length]
            bear_order_blocks.append({
                'top': top,
                'btm': btm,
                'avg': avg,
                'left': left,
                'detection_index': i
            })

    # Check for mitigation
    current_target_bull = data['target_bull'].iloc[i]
    current_target_bear = data['target_bear'].iloc[i]
    bull_order_blocks = [
        ob for ob in bull_order_blocks if current_target_bull >= ob['btm']]
    bear_order_blocks = [
        ob for ob in bear_order_blocks if current_target_bear <= ob['top']]

print(f"Bullish Order Blocks Remaining: {len(bull_order_blocks)}")
print(f"Bearish Order Blocks Remaining: {len(bear_order_blocks)}")

# --- Limit to last 'ext_last' unmitigated order blocks ---
unmitigated_bull_order_blocks = bull_order_blocks[-bull_ext_last:]
unmitigated_bear_order_blocks = bear_order_blocks[-bear_ext_last:]

# --- Plot the Chart ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['close'], label='Close', color='blue')

for ob in unmitigated_bull_order_blocks:
    left = ob['left']
    top = ob['top']
    btm = ob['btm']
    avg = ob['avg']
    right = data.index[-1]
    ax.fill_betweenx(y=[btm, top], x1=left, x2=right,
                     color=bg_bull_css, alpha=0.2, edgecolor=bull_css)
    ax.hlines(y=avg, xmin=left, xmax=right, color=bull_avg_css, linestyle='--')

for ob in unmitigated_bear_order_blocks:
    left = ob['left']
    top = ob['top']
    btm = ob['btm']
    avg = ob['avg']
    right = data.index[-1]
    ax.fill_betweenx(y=[btm, top], x1=left, x2=right,
                     color=bg_bear_css, alpha=0.2, edgecolor=bear_css)
    ax.hlines(y=avg, xmin=left, xmax=right, color=bear_avg_css, linestyle='--')

ax.set_title('BTC-USD Order Block Detector')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(['Close'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Step 3: Information about the order blocks ---
print(f"Bullish Order Blocks: {len(unmitigated_bull_order_blocks)}")
print(f"Bearish Order Blocks: {len(unmitigated_bear_order_blocks)}")
for ob in unmitigated_bull_order_blocks:
    print(ob)

# --- Step 4: Calculate Order Block Metrics ---
order_block_metrics = []

# Tính toán Order Block Volume và Percentage theo cách của smc.py
lookback = 20  # Khoảng thời gian để so sánh khối lượng
volume_ma = data['volume'].rolling(lookback).mean()

# Tạo các cột mới cho Order Block
data['ob'] = np.nan  # 1 cho bullish OB, -1 cho bearish OB
data['top'] = np.nan  # Đỉnh của order block
data['bottom'] = np.nan  # Đáy của order block
data['obVolume'] = np.nan  # Khối lượng của order block
data['percentage'] = np.nan  # Phần trăm sức mạnh của order block
data['mitigated_index'] = np.nan  # Chỉ số khi order block bị vô hiệu hóa

# Xử lý bullish order blocks
for i, ob in enumerate(unmitigated_bull_order_blocks):
    idx = data.index.get_loc(ob['left'])

    # Đánh dấu là bullish order block
    data.loc[ob['left'], 'ob'] = 1
    data.loc[ob['left'], 'top'] = ob['top']
    data.loc[ob['left'], 'bottom'] = ob['btm']

    # Tính toán khối lượng OB (volume + 2 khối lượng gần nhất)
    if idx + 2 < len(data):
        # Tính tổng khối lượng của 3 cây nến
        obVolume = (
            data['volume'].iloc[idx] +
            data['volume'].iloc[idx + 1] +
            data['volume'].iloc[idx + 2]
        )
        data.loc[ob['left'], 'obVolume'] = obVolume

        # Tính toán lowVolume và highVolume
        lowVolume = data['volume'].iloc[idx]
        highVolume = data['volume'].iloc[idx + 1] + \
            data['volume'].iloc[idx + 2]

        # Tính toán percentage theo cách của smc.py
        if idx >= lookback and volume_ma.iloc[idx] > 0:
            # Cách 1: Dựa trên khối lượng trung bình
            percentage = min(int((obVolume / volume_ma.iloc[idx]) * 100), 100)

            # Cách 2: Dựa trên tỷ lệ giữa lowVolume và highVolume
            # percentage_alt = (min(highVolume, lowVolume) / max(highVolume, lowVolume) * 100
            #                  if max(highVolume, lowVolume) != 0 else 100)

            data.loc[ob['left'], 'percentage'] = percentage

# Xử lý bearish order blocks
for i, ob in enumerate(unmitigated_bear_order_blocks):
    idx = data.index.get_loc(ob['left'])

    # Đánh dấu là bearish order block
    data.loc[ob['left'], 'ob'] = -1
    data.loc[ob['left'], 'top'] = ob['top']
    data.loc[ob['left'], 'bottom'] = ob['btm']

    # Tính toán khối lượng OB (volume + 2 khối lượng gần nhất)
    if idx + 2 < len(data):
        # Tính tổng khối lượng của 3 cây nến
        obVolume = (
            data['volume'].iloc[idx] +
            data['volume'].iloc[idx + 1] +
            data['volume'].iloc[idx + 2]
        )
        data.loc[ob['left'], 'obVolume'] = obVolume

        # Tính toán lowVolume và highVolume
        lowVolume = data['volume'].iloc[idx + 1] + data['volume'].iloc[idx + 2]
        highVolume = data['volume'].iloc[idx]

        # Tính toán percentage theo cách của smc.py
        if idx >= lookback and volume_ma.iloc[idx] > 0:
            # Cách 1: Dựa trên khối lượng trung bình
            percentage = min(int((obVolume / volume_ma.iloc[idx]) * 100), 100)

            # Cách 2: Dựa trên tỷ lệ giữa lowVolume và highVolume
            # percentage_alt = (min(highVolume, lowVolume) / max(highVolume, lowVolume) * 100
            #                  if max(highVolume, lowVolume) != 0 else 100)

            data.loc[ob['left'], 'percentage'] = percentage

# In thông tin về Order Blocks
print("\n--- Order Block Metrics ---")
ob_data = data.dropna(subset=['ob'])
if not ob_data.empty:
    for idx, row in ob_data.iterrows():
        ob_type = "Bullish" if row['ob'] == 1 else "Bearish"
        print(f"{ob_type} Order Block at {idx}:")
        print(f"  Top: {row['top']:.2f}, Bottom: {row['bottom']:.2f}")
        print(
            f"  Volume: {row['obVolume']:.2f}, Strength: {row['percentage']:.0f}%")
        print("---")
else:
    print("No active order blocks detected")
