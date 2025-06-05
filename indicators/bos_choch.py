import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from binance.client import Client
from typing import List, Dict, Optional

from binance_data_fetcher import BinanceDataFetcher


class BOSCHoCHEvent:
    """
    Class to represent a BOS/CHoCH event (Break of Structure or Change of Character)
    """

    def __init__(self, index, price, event_type, direction, time):
        self.index = index
        self.price = price
        self.event_type = event_type  # 'BOS' or 'CHoCH'
        self.direction = direction    # 1 for bullish, -1 for bearish
        self.time = time

    def __str__(self):
        return f"{self.event_type} ({'Bull' if self.direction == 1 else 'Bear'}) at {self.time}: {self.price:.2f}"


def detect_fractals(df: pd.DataFrame, length: int = 5) -> pd.DataFrame:
    """
    Detect bullish and bearish fractals in the price data.
    Returns a DataFrame with columns 'bull_fractal' and 'bear_fractal'.
    """
    p = length // 2
    df = df.copy()
    df['bull_fractal'] = False
    df['bear_fractal'] = False
    for i in range(p, len(df) - p):
        high_slice = df['high'].iloc[i-p:i+p+1]
        low_slice = df['low'].iloc[i-p:i+p+1]
        if df['high'].iloc[i] == high_slice.max() and high_slice[p] == high_slice.max():
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, p+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, p+1)):
                df.at[df.index[i], 'bull_fractal'] = True
        if df['low'].iloc[i] == low_slice.min() and low_slice[p] == low_slice.min():
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, p+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, p+1)):
                df.at[df.index[i], 'bear_fractal'] = True
    return df


def detect_bos_choch(df: pd.DataFrame, length: int = 5) -> List[BOSCHoCHEvent]:
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character) events based on fractals and close price crossing fractal levels.
    Returns a list of BOSCHoCHEvent objects.
    """
    df = detect_fractals(df, length)
    p = length // 2
    events = []
    upper = {'value': None, 'loc': None, 'iscrossed': False}
    lower = {'value': None, 'loc': None, 'iscrossed': False}
    os = 0  # 1 for bullish, -1 for bearish, 0 for neutral
    for n in range(len(df)):
        # Bullish fractal
        if df['bull_fractal'].iloc[n]:
            upper['value'] = df['high'].iloc[n]
            upper['loc'] = n
            upper['iscrossed'] = False
        # Bearish fractal
        if df['bear_fractal'].iloc[n]:
            lower['value'] = df['low'].iloc[n]
            lower['loc'] = n
            lower['iscrossed'] = False
        # Bullish BOS/CHoCH
        if upper['value'] is not None and not upper['iscrossed']:
            if df['close'].iloc[n] > upper['value']:
                event_type = 'CHoCH' if os == -1 else 'BOS'
                events.append(BOSCHoCHEvent(
                    index=n,
                    price=upper['value'],
                    event_type=event_type,
                    direction=1,
                    time=df.index[n]
                ))
                upper['iscrossed'] = True
                os = 1
        # Bearish BOS/CHoCH
        if lower['value'] is not None and not lower['iscrossed']:
            if df['close'].iloc[n] < lower['value']:
                event_type = 'CHoCH' if os == 1 else 'BOS'
                events.append(BOSCHoCHEvent(
                    index=n,
                    price=lower['value'],
                    event_type=event_type,
                    direction=-1,
                    time=df.index[n]
                ))
                lower['iscrossed'] = True
                os = -1
    return events


def plot_bos_choch(df: pd.DataFrame, events: List[BOSCHoCHEvent], symbol: str = '', interval: str = ''):
    """
    Plot price chart with BOS/CHoCH events, similar to TradingView.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df.index, df['close'], label='Close', color='black', linewidth=1)
    for event in events:
        color = '#089981' if event.direction == 1 else '#f23645'
        label = f"{event.event_type}"
        va = 'top' if event.direction == 1 else 'bottom'
        ax.scatter(df.index[event.index], event.price, color=color,
                   marker='^' if event.direction == 1 else 'v', s=80, zorder=5)
        ax.text(df.index[event.index], event.price, label, color=color,
                fontsize=9, va=va, ha='center', fontweight='bold')
    ax.set_title(f"{symbol} {interval} BOS/CHoCH Structure", fontsize=16)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.2)
    plt.legend(['Close'])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='BOS/CHoCH Market Structure Detector')
    parser.add_argument('--symbol', type=str,
                        default='SOLUSDT', help='Symbol to fetch data')
    parser.add_argument('--interval', type=str, default='15m',
                        help='Interval to fetch data')
    parser.add_argument('--length', type=int, default=5,
                        help='Fractal length (window)')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days of historical data')
    args = parser.parse_args()
    client = Client()
    fetcher = BinanceDataFetcher(client)
    start_time = datetime.now() - timedelta(days=args.days)
    df = fetcher.get_historical_klines(args.symbol, args.interval, start_time)
    events = detect_bos_choch(df, length=args.length)
    print(f"Detected {len(events)} BOS/CHoCH events.")
    for event in events:
        print(event)
    plot_bos_choch(df, events, symbol=args.symbol, interval=args.interval)
