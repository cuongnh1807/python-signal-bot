import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
from indicators.bos_choch import detect_bos_choch, plot_bos_choch
from binance_data_fetcher import BinanceDataFetcher


def test_bos_choch_solusdt():
    symbol = 'SOLUSDT'
    interval = '15m'
    days = 7
    print(f"\n🧪 Testing BOS/CHoCH on {symbol} {interval}...")
    client = Client()
    fetcher = BinanceDataFetcher(client)
    start_time = datetime.now() - timedelta(days=days)
    df = fetcher.get_historical_klines(symbol, interval, start_time)
    print(
        f"✅ Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    events = detect_bos_choch(df, length=5)
    print(f"Detected {len(events)} BOS/CHoCH events.")
    for event in events:
        print(event)
    plot_bos_choch(df, events, symbol=symbol, interval=interval)


if __name__ == "__main__":
    test_bos_choch_solusdt()
