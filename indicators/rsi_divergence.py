import pandas as pd
import numpy as np
from strategy import calculate_rsi


def find_pivot_points(data, left_bars, right_bars):
    """Find pivot highs and lows in the data"""
    pivot_highs = []
    pivot_lows = []

    for i in range(left_bars, len(data) - right_bars):
        window = data[i-left_bars:i+right_bars+1]
        if data[i] == max(window):
            pivot_highs.append(i)
        if data[i] == min(window):
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


def check_regular_bullish_divergence(price, rsi, pivot_idx, lookback_range):
    """Check for regular bullish divergence"""
    divergence_points = []

    for i in range(len(pivot_idx)-1):
        curr_idx = pivot_idx[i]
        prev_idx = pivot_idx[i-1]

        if lookback_range[0] <= curr_idx - prev_idx <= lookback_range[1]:
            # Price: Lower Low
            if price[curr_idx] < price[prev_idx]:
                # RSI: Higher Low
                if rsi[curr_idx] > rsi[prev_idx]:
                    divergence_points.append((prev_idx, curr_idx))
    return divergence_points


def check_hidden_bullish_divergence(price, rsi, pivot_idx, lookback_range):
    """Check for hidden bullish divergence"""
    divergence_points = []

    for i in range(len(pivot_idx)-1):
        curr_idx = pivot_idx[i]
        prev_idx = pivot_idx[i-1]

        if lookback_range[0] <= curr_idx - prev_idx <= lookback_range[1]:
            # Price: Higher Low
            if price[curr_idx] > price[prev_idx]:
                # RSI: Lower Low
                if rsi[curr_idx] < rsi[prev_idx]:
                    divergence_points.append((prev_idx, curr_idx))
    return divergence_points


def check_regular_bearish_divergence(price, rsi, pivot_idx, lookback_range):
    """Check for regular bearish divergence"""
    divergence_points = []

    for i in range(len(pivot_idx)-1):
        curr_idx = pivot_idx[i]
        prev_idx = pivot_idx[i-1]

        if lookback_range[0] <= curr_idx - prev_idx <= lookback_range[1]:
            # Price: Higher High
            if price[curr_idx] > price[prev_idx]:
                # RSI: Lower High
                if rsi[curr_idx] < rsi[prev_idx]:
                    divergence_points.append((prev_idx, curr_idx))
    return divergence_points


def check_hidden_bearish_divergence(price, rsi, pivot_idx, lookback_range):
    """Check for hidden bearish divergence"""
    divergence_points = []

    for i in range(len(pivot_idx)-1):
        curr_idx = pivot_idx[i]
        prev_idx = pivot_idx[i-1]

        if lookback_range[0] <= curr_idx - prev_idx <= lookback_range[1]:
            # Price: Lower High
            if price[curr_idx] < price[prev_idx]:
                # RSI: Higher High
                if rsi[curr_idx] > rsi[prev_idx]:
                    divergence_points.append((prev_idx, curr_idx))
    return divergence_points


def find_rsi_divergences(df, rsi_period=14, left_bars=5, right_bars=5,
                         range_lower=5, range_upper=60):
    """Main function to find RSI divergences"""

    # Tính RSI

    rsi = calculate_rsi(df)

    # Tìm các pivot points
    pivot_highs, pivot_lows = find_pivot_points(rsi, left_bars, right_bars)

    lookback_range = (range_lower, range_upper)

    divergences = {
        'regular_bullish': check_regular_bullish_divergence(
            df['low'], rsi, pivot_lows, lookback_range
        ),
        'hidden_bullish': check_hidden_bullish_divergence(
            df['low'], rsi, pivot_lows, lookback_range
        ),
        'regular_bearish': check_regular_bearish_divergence(
            df['high'], rsi, pivot_highs, lookback_range
        ),
        'hidden_bearish': check_hidden_bearish_divergence(
            df['high'], rsi, pivot_highs, lookback_range
        )
    }

    return divergences

# Sử dụng:
# df = pd.DataFrame với các cột: open, high, low, close
# divergences = find_rsi_divergences(df)
