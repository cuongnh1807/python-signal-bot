#!/usr/bin/env python3

from binance.client import Client
from binance_data_fetcher import BinanceDataFetcher
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to path to import our indicators
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def analyze_fvg_with_different_thresholds():
    """Analyze FVG detection with different threshold values to find the right balance"""

    print("🔍 ANALYZING FVG FILTERING TO MATCH TRADINGVIEW")
    print("=" * 60)

    # Fetch data - same timeframe as shown in TradingView image (about 6 days)
    client = Client()
    data_fetcher = BinanceDataFetcher(client=client)

    start_time = datetime.now() - timedelta(days=6)
    df = data_fetcher.get_historical_klines(
        symbol='BTCUSDT',
        interval='15m',
        start_time=start_time
    )

    if df.empty:
        print("❌ No data fetched")
        return

    print(f"📊 Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Test different thresholds
    thresholds_to_test = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    print(f"\n🧪 TESTING DIFFERENT MANUAL THRESHOLDS:")
    print("Threshold% | Total FVGs | Bull | Bear | Avg Gap Size%")
    print("-" * 55)

    results = []

    for threshold_pct in thresholds_to_test:
        fvgs = detect_fvgs_with_threshold(df, threshold_pct / 100)

        bull_count = len([fvg for fvg in fvgs if fvg['is_bullish']])
        bear_count = len([fvg for fvg in fvgs if not fvg['is_bullish']])

        if fvgs:
            avg_gap_size = np.mean([fvg['gap_size_pct'] for fvg in fvgs])
        else:
            avg_gap_size = 0

        print(f"{threshold_pct:9.2f}% | {len(fvgs):9d} | {bull_count:4d} | {bear_count:4d} | {avg_gap_size:11.4f}%")

        results.append({
            'threshold': threshold_pct,
            'total': len(fvgs),
            'bull': bull_count,
            'bear': bear_count,
            'avg_gap': avg_gap_size
        })

    # Find optimal threshold (aim for 10-20 FVGs total to match TradingView)
    optimal_threshold = None
    for result in results:
        if 10 <= result['total'] <= 25:  # Target range based on TradingView image
            optimal_threshold = result['threshold']
            break

    if optimal_threshold:
        print(f"\n✅ RECOMMENDED THRESHOLD: {optimal_threshold}%")
        print(
            f"   This produces {[r for r in results if r['threshold'] == optimal_threshold][0]['total']} FVGs")
    else:
        print(f"\n💡 Consider threshold between 0.05% - 0.15% for optimal results")

    # Test automatic threshold
    print(f"\n🤖 TESTING AUTOMATIC THRESHOLD:")
    auto_fvgs = detect_fvgs_with_auto_threshold(df)
    auto_bull = len([fvg for fvg in auto_fvgs if fvg['is_bullish']])
    auto_bear = len([fvg for fvg in auto_fvgs if not fvg['is_bullish']])

    if auto_fvgs:
        auto_avg_gap = np.mean([fvg['gap_size_pct'] for fvg in auto_fvgs])
        auto_threshold_used = auto_fvgs[0]['threshold_used'] * 100
    else:
        auto_avg_gap = 0
        auto_threshold_used = 0

    print(f"Auto Mode  | {len(auto_fvgs):9d} | {auto_bull:4d} | {auto_bear:4d} | {auto_avg_gap:11.4f}% (threshold: {auto_threshold_used:.4f}%)")

    return results, optimal_threshold


def detect_fvgs_with_threshold(df: pd.DataFrame, threshold: float):
    """Detect FVGs with manual threshold"""
    df_reset = df.reset_index(drop=True)
    original_index = df.index.copy()

    fvgs = []

    for i in range(2, len(df_reset)):
        current_high = df_reset.loc[i, 'high']
        current_low = df_reset.loc[i, 'low']
        prev_close = df_reset.loc[i-1, 'close']
        prev2_high = df_reset.loc[i-2, 'high']
        prev2_low = df_reset.loc[i-2, 'low']

        # Bullish FVG
        if (current_low > prev2_high and
                prev_close > prev2_high):

            gap_size_pct = ((current_low - prev2_high) / prev2_high) * 100

            if gap_size_pct > threshold * 100:  # Convert to percentage for comparison
                fvgs.append({
                    'is_bullish': True,
                    'time': original_index[i],
                    'top': current_low,
                    'bottom': prev2_high,
                    'gap_size_pct': gap_size_pct,
                    'threshold_used': threshold
                })

        # Bearish FVG
        elif (current_high < prev2_low and
              prev_close < prev2_low):

            gap_size_pct = ((prev2_low - current_high) / current_high) * 100

            if gap_size_pct > threshold * 100:  # Convert to percentage for comparison
                fvgs.append({
                    'is_bullish': False,
                    'time': original_index[i],
                    'top': prev2_low,
                    'bottom': current_high,
                    'gap_size_pct': gap_size_pct,
                    'threshold_used': threshold
                })

    return fvgs


def detect_fvgs_with_auto_threshold(df: pd.DataFrame):
    """Detect FVGs with automatic threshold like Pine Script"""
    df_reset = df.reset_index(drop=True)
    original_index = df.index.copy()

    # Calculate automatic threshold exactly like Pine Script
    relative_ranges = (df_reset['high'] - df_reset['low']) / df_reset['low']
    cumulative_sum = relative_ranges.cumsum()
    bar_indices = np.arange(1, len(df_reset) + 1)
    auto_threshold_series = cumulative_sum / bar_indices

    fvgs = []

    for i in range(2, len(df_reset)):
        current_high = df_reset.loc[i, 'high']
        current_low = df_reset.loc[i, 'low']
        prev_close = df_reset.loc[i-1, 'close']
        prev2_high = df_reset.loc[i-2, 'high']
        prev2_low = df_reset.loc[i-2, 'low']
        threshold = auto_threshold_series.iloc[i]

        # Bullish FVG
        if (current_low > prev2_high and
                prev_close > prev2_high):

            gap_size = (current_low - prev2_high) / prev2_high

            if gap_size > threshold:
                fvgs.append({
                    'is_bullish': True,
                    'time': original_index[i],
                    'top': current_low,
                    'bottom': prev2_high,
                    'gap_size_pct': gap_size * 100,
                    'threshold_used': threshold
                })

        # Bearish FVG
        elif (current_high < prev2_low and
              prev_close < prev2_low):

            gap_size = (prev2_low - current_high) / current_high

            if gap_size > threshold:
                fvgs.append({
                    'is_bullish': False,
                    'time': original_index[i],
                    'top': prev2_low,
                    'bottom': current_high,
                    'gap_size_pct': gap_size * 100,
                    'threshold_used': threshold
                })

    return fvgs


def analyze_gap_sizes():
    """Analyze the distribution of gap sizes to understand what should be filtered"""

    print(f"\n📈 ANALYZING GAP SIZE DISTRIBUTION:")
    print("=" * 50)

    # Get all potential gaps (threshold = 0)
    client = Client()
    data_fetcher = BinanceDataFetcher(client=client)

    start_time = datetime.now() - timedelta(days=6)
    df = data_fetcher.get_historical_klines(
        symbol='BTCUSDT',
        interval='15m',
        start_time=start_time
    )

    all_gaps = detect_fvgs_with_threshold(df, 0.0)

    if not all_gaps:
        print("No gaps found")
        return

    gap_sizes = [gap['gap_size_pct'] for gap in all_gaps]
    gap_sizes.sort()

    print(f"Total potential gaps: {len(gap_sizes)}")
    print(f"Min gap size: {min(gap_sizes):.4f}%")
    print(f"Max gap size: {max(gap_sizes):.4f}%")
    print(f"Mean gap size: {np.mean(gap_sizes):.4f}%")
    print(f"Median gap size: {np.median(gap_sizes):.4f}%")
    print(f"95th percentile: {np.percentile(gap_sizes, 95):.4f}%")
    print(f"90th percentile: {np.percentile(gap_sizes, 90):.4f}%")
    print(f"75th percentile: {np.percentile(gap_sizes, 75):.4f}%")

    # Show distribution
    print(f"\n📊 GAP SIZE DISTRIBUTION:")
    ranges = [
        (0, 0.05, "0.00-0.05%"),
        (0.05, 0.1, "0.05-0.10%"),
        (0.1, 0.2, "0.10-0.20%"),
        (0.2, 0.5, "0.20-0.50%"),
        (0.5, 1.0, "0.50-1.00%"),
        (1.0, float('inf'), ">1.00%")
    ]

    for min_val, max_val, label in ranges:
        count = len([g for g in gap_sizes if min_val <= g < max_val])
        percentage = (count / len(gap_sizes)) * 100
        print(f"{label:12s}: {count:3d} gaps ({percentage:5.1f}%)")


if __name__ == "__main__":
    print("🚀 FVG FILTERING ANALYSIS TO MATCH TRADINGVIEW")
    print("=" * 60)

    # Step 1: Analyze gap size distribution
    analyze_gap_sizes()

    # Step 2: Test different thresholds
    results, optimal = analyze_fvg_with_different_thresholds()

    print(f"\n💡 CONCLUSIONS:")
    print("=" * 30)
    print("• TradingView appears to use a higher threshold than 0.0%")
    print("• Optimal threshold seems to be around 0.05% - 0.15%")
    print("• This filters out noise while keeping significant gaps")
    print("• Auto threshold might actually be used in TradingView")

    if optimal:
        print(
            f"\n🎯 RECOMMENDATION: Use threshold {optimal}% for TradingView-like results")
