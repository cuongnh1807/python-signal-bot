import pandas as pd
import matplotlib.pyplot as plt
from helpers.price import detect_trend_from_ema
from live_trading_bot import LiveTradingBot, TelegramNotifier
import mplfinance as mpf
from binance.client import Client
import os
from dotenv import load_dotenv

from smartmoneyconcepts import smc

# Load environment variables
load_dotenv()

# Khởi tạo Binance client
client = Client(
    api_key=os.getenv('BINANCE_API_KEY'),
    api_secret=os.getenv('BINANCE_API_SECRET')
)

# Khởi tạo bot
bot = LiveTradingBot(
    client=client,
    symbol='BTCUSDT',
    interval='15m',
    max_risk_per_trade=0.02,
    leverage=20,
    window_size=1000,
    min_setup_quality=10,
    min_volume_ratio=1,
    test_mode=True,
    telegram=None  # Không cần telegram cho test
)

bot._fetch_latest_data()

df = bot.historical_data
# result = detect_trend_from_ema(df)
# print(result)
# Vẽ biểu đồ


# def plot_with_orderblocks(df, orderblocks):
#     # Tạo plot style
#     style = mpf.make_mpf_style(base_mpf_style='charles', rc={
#                                'figure.figsize': (15, 8)})

#     # Tạo các box cho orderblocks
#     boxes = []
#     colors = []
#     for ob in orderblocks:
#         # Tạo box cho mỗi orderblock
#         box = dict(
#             y1=ob['low'],
#             y2=ob['high'],
#             x1=ob['start_time'],
#             x2=ob['end_time'],
#             alpha=0.3
#         )
#         boxes.append(box)
#         # Màu xanh cho bullish, đỏ cho bearish
#         colors.append('g' if ob['direction'] == 1 else 'r')

#     # Vẽ biểu đồ với orderblocks
#     mpf.plot(
#         df,
#         type='candle',
#         style=style,
#         title='SOLUSDT Orderblocks',
#         alines=dict(alines=boxes, colors=colors, alpha=0.3, linewidths=7),
#         volume=True,
#         show_nontrading=True
#     )


# # Vẽ biểu đồ
# plot_with_orderblocks(df, orderblocks)
# plt.show()
