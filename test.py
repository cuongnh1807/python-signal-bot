import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from binance_data_fetcher import BinanceDataFetcher
from smartmoneyconcepts.smc import smc
from strategy import analyze_trading_setup

# Khởi tạo fetcher dữ liệu
fetcher = BinanceDataFetcher()

# Lấy dữ liệu BTC từ Binance
symbol = "BTCUSDT"
interval = "15m"  # Khoảng thời gian 1 giờ
start_time = datetime.now() - timedelta(days=5)  # Dữ liệu 30 ngày gần nhất
data = fetcher.get_historical_klines(symbol, interval, start_time)

# Tính toán swing high/low
swing_hl = smc.swing_highs_lows(data, swing_length=10)

# Phân tích order blocks
ob_results = smc.ob(data, swing_hl)

# Phân tích setup giao dịch
analysis_results = analyze_trading_setup(data, swing_hl)

# Vẽ biểu đồ
plt.figure(figsize=(16, 10))

# Vẽ nến


def plot_candlestick(ax, data):
    width = 0.6
    width2 = 0.05

    # Tạo mảng cho trục x
    t = np.arange(len(data.index))

    # Vẽ nến tăng (xanh)
    up = data[data.close >= data.open]
    ax.bar(t[data.index.isin(up.index)], up.close - up.open,
           width, bottom=up.open, color='green', alpha=0.5)
    ax.bar(t[data.index.isin(up.index)], up.high - up.close,
           width2, bottom=up.close, color='green', alpha=0.5)
    ax.bar(t[data.index.isin(up.index)], up.open - up.low,
           width2, bottom=up.low, color='green', alpha=0.5)

    # Vẽ nến giảm (đỏ)
    down = data[data.close < data.open]
    ax.bar(t[data.index.isin(down.index)], down.open - down.close,
           width, bottom=down.close, color='red', alpha=0.5)
    ax.bar(t[data.index.isin(down.index)], down.high - down.open,
           width2, bottom=down.open, color='red', alpha=0.5)
    ax.bar(t[data.index.isin(down.index)], down.close - down.low,
           width2, bottom=down.low, color='red', alpha=0.5)

    # Định dạng trục x để hiển thị ngày tháng
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xticks(t[::24])  # Hiển thị mỗi ngày (24 giờ)
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in data.index[::24]])
    plt.xticks(rotation=45)

    return t


# Tạo subplot
ax = plt.subplot(1, 1, 1)
t = plot_candlestick(ax, data)

# Vẽ Order Blocks
for i in range(len(ob_results)):
    if pd.notna(ob_results["OB"][i]):
        ob_direction = ob_results["OB"][i]
        ob_top = ob_results["Top"][i]
        ob_bottom = ob_results["Bottom"][i]

        # Vẽ hình chữ nhật cho Order Block
        color = 'green' if ob_direction == 1 else 'red'

        # Tìm vị trí của order block trên trục x
        x_pos = t[i]

        # Vẽ order block từ vị trí hiện tại đến cuối biểu đồ
        rect = plt.Rectangle((x_pos, ob_bottom), len(t) - x_pos, ob_top - ob_bottom,
                             color=color, alpha=0.2)
        ax.add_patch(rect)

        # Thêm nhãn
        plt.text(x_pos, ob_top, f"OB {'Bull' if ob_direction == 1 else 'Bear'}",
                 fontsize=8, color=color)

# Vẽ swing high/low
for i in range(len(swing_hl)):
    if pd.notna(swing_hl["HighLow"][i]):
        if swing_hl["HighLow"][i] == 1:  # Swing High
            plt.scatter(t[i], swing_hl["Level"][i],
                        color='blue', marker='^', s=100)
        else:  # Swing Low
            plt.scatter(t[i], swing_hl["Level"][i],
                        color='purple', marker='v', s=100)

# Thêm thông tin về setup giao dịch
if analysis_results["trade_setups"]:
    setup_info = analysis_results["trade_setups"][0]
    setup_text = (
        f"Setup: {setup_info['setup_type']}\n"
        f"Chất lượng: {setup_info['setup_quality']:.1f}%\n"
        f"Khuyến nghị: {setup_info['trade_recommendation']}"
    )
    plt.figtext(0.02, 0.02, setup_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

# Thêm tiêu đề và nhãn
plt.title(f'BTC/USDT {interval} với Order Blocks và Swing Points')
plt.ylabel('Giá (USDT)')
plt.grid(True, alpha=0.3)

# Hiển thị biểu đồ
plt.tight_layout()
plt.savefig('btc_orderblocks.png')
plt.show()

# In thông tin về các setup giao dịch
print("\nCác setup giao dịch được phát hiện:")
for i, setup in enumerate(analysis_results["trade_setups"]):
    print(f"\nSetup #{i+1}:")
    print(f"Loại: {setup['setup_type']} - {setup['setup_description']}")
    print(f"Chất lượng setup: {setup['setup_quality']:.1f}%")
    print(f"Khuyến nghị: {setup['trade_recommendation']}")
    print(f"Mức OB: {setup['ob_level']}")
    print(f"Điểm mạnh: {setup['setup_strength']}")
    if setup['warning_messages']:
        print("Cảnh báo:")
        for warning in setup['warning_messages']:
            print(f"  - {warning}")
