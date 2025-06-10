import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from binance.client import Client
from typing import List, Optional, Dict, Tuple

from binance_data_fetcher import BinanceDataFetcher


class OBSwing:
    """Class to represent swing points for order block detection"""

    def __init__(self, x: int = None, y: float = None, swing_volume: float = None, crossed: bool = False):
        self.x = x  # bar index
        self.y = y  # price
        self.swing_volume = swing_volume
        self.crossed = crossed


class OrderBlockInfo:
    """Enhanced order block information with breaker logic"""

    def __init__(self, top: float, bottom: float, ob_volume: float, ob_type: str,
                 start_time: datetime, start_index: int = None):
        self.top = top
        self.bottom = bottom
        self.ob_volume = ob_volume
        self.ob_type = ob_type  # "Bull" or "Bear"
        self.start_time = start_time
        self.start_index = start_index

        # Volume breakdown
        self.ob_low_volume = None
        self.ob_high_volume = None
        self.bb_volume = None  # Break volume

        # Breaker status
        self.breaker = False
        self.break_time = None
        self.break_index = None
        self.disabled = False

        # Additional properties
        self.timeframe_str = None
        self.combined = False
        self.combined_timeframes_str = None

        # Entry evaluation (if available)
        self.entry_score = None
        self.entry_quality = None
        self.risk_level = None
        self.warnings = []
        self.trend_confluence = None

    def get_area(self, current_time: datetime) -> float:
        """Calculate area of the order block"""
        end_time = self.break_time if self.break_time else current_time
        if end_time <= self.start_time:
            width = 1
        else:
            width = (end_time - self.start_time).total_seconds()
        height = abs(self.top - self.bottom)
        return width * height

    def copy(self):
        """Create a copy of the order block"""
        new_ob = OrderBlockInfo(
            self.top, self.bottom, self.ob_volume,
            self.ob_type, self.start_time, self.start_index
        )
        new_ob.ob_low_volume = self.ob_low_volume
        new_ob.ob_high_volume = self.ob_high_volume
        new_ob.bb_volume = self.bb_volume
        new_ob.breaker = self.breaker
        new_ob.break_time = self.break_time
        new_ob.break_index = self.break_index
        new_ob.disabled = self.disabled
        new_ob.timeframe_str = self.timeframe_str
        new_ob.combined = self.combined
        new_ob.combined_timeframes_str = self.combined_timeframes_str
        return new_ob


def find_ob_swings(df: pd.DataFrame, swing_length: int) -> Tuple[List[OBSwing], List[OBSwing]]:
    """
    Find swing highs and lows using Pine Script logic
    """
    swing_type = 0
    top_swings = []
    bottom_swings = []

    # Calculate rolling highest and lowest
    df['upper'] = df['high'].rolling(window=swing_length).max()
    df['lower'] = df['low'].rolling(window=swing_length).min()

    for i in range(swing_length, len(df)):
        prev_swing_type = swing_type

        # Pine Script logic: swingType := high[len] > upper ? 0 : low[len] < lower ? 1 : swingType
        if df.iloc[i - swing_length]['high'] > df.iloc[i]['upper']:
            swing_type = 0  # High swing
        elif df.iloc[i - swing_length]['low'] < df.iloc[i]['lower']:
            swing_type = 1  # Low swing

        # Create swing points when type changes
        if swing_type == 0 and prev_swing_type != 0:
            top_swings.append(OBSwing(
                x=i - swing_length,
                y=df.iloc[i - swing_length]['high'],
                swing_volume=df.iloc[i - swing_length]['volume']
            ))

        if swing_type == 1 and prev_swing_type != 1:
            bottom_swings.append(OBSwing(
                x=i - swing_length,
                y=df.iloc[i - swing_length]['low'],
                swing_volume=df.iloc[i - swing_length]['volume']
            ))

    return top_swings, bottom_swings


def detect_order_breaker_blocks(df: pd.DataFrame, swing_length: int = 10, max_atr_mult: float = 3.5,
                                max_distance_to_last_bar: int = 1750, max_order_blocks: int = 30,
                                ob_end_method: str = "Wick", bullish_ob_count: int = 10,
                                bearish_ob_count: int = 10, combine_obs: bool = True,
                                overlap_threshold: float = 0.0, use_entry_evaluation: bool = True,
                                entry_threshold: int = 45) -> List[OrderBlockInfo]:
    """
    Detect order blocks with breaker logic based on Pine Script implementation
    """
    # Store original index for datetime reference
    original_index = df.index.copy()
    df_work = df.reset_index(drop=True).copy()

    # Calculate ATR for size filtering
    df_work['tr'] = np.maximum(
        df_work['high'] - df_work['low'],
        np.maximum(
            abs(df_work['high'] - df_work['close'].shift(1)),
            abs(df_work['low'] - df_work['close'].shift(1))
        )
    )
    df_work['atr'] = df_work['tr'].rolling(10).mean()
    df_work['atr'] = df_work['atr'].fillna(df_work['tr'].mean())

    # Initialize order block lists
    bullish_order_blocks = []
    bearish_order_blocks = []
    all_order_blocks = []  # For tracking all OBs including broken ones

    # Find swings
    top_swings, bottom_swings = find_ob_swings(df_work, swing_length)

    print(
        f"Found {len(top_swings)} top swings and {len(bottom_swings)} bottom swings")

    # Process each bar (Pine Script: if bar_index > last_bar_index - maxDistanceToLastBar)
    last_bar_index = len(df_work) - 1

    for bar_idx in range(len(df_work)):
        if bar_idx <= last_bar_index - max_distance_to_last_bar:
            continue

        current_bar = df_work.iloc[bar_idx]
        current_close = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_volume = current_bar['volume']
        current_atr = current_bar['atr']

        # Check mitigation for existing bullish order blocks
        for i, ob in enumerate(bullish_order_blocks[:]):
            if not ob.breaker:
                mitigation_price = current_low if ob_end_method == "Wick" else min(
                    df_work.iloc[bar_idx]['open'], current_close)
                if mitigation_price < ob.bottom:
                    ob.breaker = True
                    ob.break_time = original_index[bar_idx]
                    ob.break_index = bar_idx
                    ob.bb_volume = current_volume
                    print(
                        f"Bullish OB broken at {ob.break_time}: {ob.bottom:.4f} - {ob.top:.4f}")
            else:
                # Remove completely broken order blocks
                if current_high > ob.top:
                    bullish_order_blocks.remove(ob)
                    print(
                        f"Removed bullish OB: {ob.bottom:.4f} - {ob.top:.4f}")

        # Check mitigation for existing bearish order blocks
        for i, ob in enumerate(bearish_order_blocks[:]):
            if not ob.breaker:
                mitigation_price = current_high if ob_end_method == "Wick" else max(
                    df_work.iloc[bar_idx]['open'], current_close)
                if mitigation_price > ob.top:
                    ob.breaker = True
                    ob.break_time = original_index[bar_idx]
                    ob.break_index = bar_idx
                    ob.bb_volume = current_volume
                    print(
                        f"Bearish OB broken at {ob.break_time}: {ob.bottom:.4f} - {ob.top:.4f}")
            else:
                # Remove completely broken order blocks
                if current_low < ob.bottom:
                    bearish_order_blocks.remove(ob)
                    print(
                        f"Removed bearish OB: {ob.bottom:.4f} - {ob.top:.4f}")

        # Check for new bullish order block creation
        for top_swing in top_swings:
            if top_swing.x >= bar_idx or top_swing.crossed:
                continue

            if current_close > top_swing.y:
                top_swing.crossed = True
                print(
                    f"Bar {bar_idx}: Bullish trigger - close {current_close:.4f} > top {top_swing.y:.4f}")

                # Pine Script order block formation logic
                if bar_idx >= 1:
                    # Initial values from previous bar
                    box_bottom = df_work.iloc[bar_idx - 1]['high']  # max[1]
                    box_top = df_work.iloc[bar_idx - 1]['low']      # min[1]
                    box_location_idx = bar_idx - 1

                    # Look back to find the actual order block
                    for i in range(1, bar_idx - top_swing.x):
                        if bar_idx - i < 0:
                            break
                        check_idx = bar_idx - i
                        current_min = df_work.iloc[check_idx]['low']
                        current_max = df_work.iloc[check_idx]['high']

                        # Pine Script logic
                        if current_min < box_top:
                            box_top = current_min
                            box_bottom = current_max
                            box_location_idx = check_idx

                    # Calculate volume breakdown
                    total_volume = current_volume
                    ob_low_volume = 0
                    ob_high_volume = current_volume

                    if bar_idx >= 1:
                        vol_1 = df_work.iloc[bar_idx - 1]['volume']
                        total_volume += vol_1
                        ob_high_volume += vol_1

                    if bar_idx >= 2:
                        vol_2 = df_work.iloc[bar_idx - 2]['volume']
                        total_volume += vol_2
                        ob_low_volume = vol_2

                    # Create order block
                    ob_info = OrderBlockInfo(
                        top=box_bottom,
                        bottom=box_top,
                        ob_volume=total_volume,
                        ob_type="Bull",
                        start_time=original_index[box_location_idx] if hasattr(original_index[box_location_idx], 'strftime')
                        else pd.to_datetime(original_index[box_location_idx])
                    )
                    ob_info.ob_low_volume = ob_low_volume
                    ob_info.ob_high_volume = ob_high_volume
                    ob_info.start_index = box_location_idx

                    # Size filter
                    ob_size = abs(ob_info.top - ob_info.bottom)
                    if ob_size <= current_atr * max_atr_mult:
                        all_order_blocks.append(ob_info)

                        # Apply entry evaluation if enabled
                        should_add = True
                        if use_entry_evaluation:
                            should_add, eval_result = evaluate_entry_quality(
                                df_work, ob_info, bar_idx, all_order_blocks)
                            if should_add:
                                # Add evaluation metrics
                                ob_info.entry_score = eval_result.get(
                                    'final_score', 50)
                                ob_info.entry_quality = eval_result.get(
                                    'entry_quality', 'Moderate')
                                ob_info.risk_level = eval_result.get(
                                    'risk_level', 'Medium')
                                ob_info.warnings = eval_result.get(
                                    'warnings', [])
                                ob_info.trend_confluence = eval_result.get(
                                    'confluence', {})

                        if should_add:
                            bullish_order_blocks.insert(0, ob_info)
                            print(
                                f"Created bullish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}, Volume: {total_volume:.0f}")

                            if len(bullish_order_blocks) > max_order_blocks:
                                bullish_order_blocks.pop()

        # Check for new bearish order block creation
        for bottom_swing in bottom_swings:
            if bottom_swing.x >= bar_idx or bottom_swing.crossed:
                continue

            if current_close < bottom_swing.y:
                bottom_swing.crossed = True
                print(
                    f"Bar {bar_idx}: Bearish trigger - close {current_close:.4f} < bottom {bottom_swing.y:.4f}")

                # Pine Script order block formation logic for bearish
                if bar_idx >= 1:
                    # Initial values from previous bar
                    box_bottom = df_work.iloc[bar_idx - 1]['low']   # min[1]
                    box_top = df_work.iloc[bar_idx - 1]['high']    # max[1]
                    box_location_idx = bar_idx - 1

                    # Look back to find the actual order block
                    for i in range(1, bar_idx - bottom_swing.x):
                        if bar_idx - i < 0:
                            break
                        check_idx = bar_idx - i
                        current_max = df_work.iloc[check_idx]['high']
                        current_min = df_work.iloc[check_idx]['low']

                        # Pine Script logic for bearish
                        if current_max > box_top:
                            box_top = current_max
                            box_bottom = current_min
                            box_location_idx = check_idx

                    # Calculate volume breakdown for bearish
                    total_volume = current_volume
                    ob_low_volume = current_volume
                    ob_high_volume = 0

                    if bar_idx >= 1:
                        vol_1 = df_work.iloc[bar_idx - 1]['volume']
                        total_volume += vol_1
                        ob_low_volume += vol_1

                    if bar_idx >= 2:
                        vol_2 = df_work.iloc[bar_idx - 2]['volume']
                        total_volume += vol_2
                        ob_high_volume = vol_2

                    # Create bearish order block
                    ob_info = OrderBlockInfo(
                        top=box_top,
                        bottom=box_bottom,
                        ob_volume=total_volume,
                        ob_type="Bear",
                        start_time=original_index[box_location_idx] if hasattr(original_index[box_location_idx], 'strftime')
                        else pd.to_datetime(original_index[box_location_idx])
                    )
                    ob_info.ob_low_volume = ob_low_volume
                    ob_info.ob_high_volume = ob_high_volume
                    ob_info.start_index = box_location_idx

                    # Size filter
                    ob_size = abs(ob_info.top - ob_info.bottom)
                    if ob_size <= current_atr * max_atr_mult:
                        all_order_blocks.append(ob_info)

                        # Apply entry evaluation if enabled
                        should_add = True
                        if use_entry_evaluation:
                            should_add, eval_result = evaluate_entry_quality(
                                df_work, ob_info, bar_idx, all_order_blocks)
                            if should_add:
                                # Add evaluation metrics
                                ob_info.entry_score = eval_result.get(
                                    'final_score', 50)
                                ob_info.entry_quality = eval_result.get(
                                    'entry_quality', 'Moderate')
                                ob_info.risk_level = eval_result.get(
                                    'risk_level', 'Medium')
                                ob_info.warnings = eval_result.get(
                                    'warnings', [])
                                ob_info.trend_confluence = eval_result.get(
                                    'confluence', {})

                        if should_add:
                            bearish_order_blocks.insert(0, ob_info)
                            print(
                                f"Created bearish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}, Volume: {total_volume:.0f}")

                            if len(bearish_order_blocks) > max_order_blocks:
                                bearish_order_blocks.pop()

    # Combine overlapping order blocks if enabled
    combined_bullish = bullish_order_blocks
    combined_bearish = bearish_order_blocks

    if combine_obs and overlap_threshold > 0:
        combined_bullish = combine_overlapping_obs(
            bullish_order_blocks, overlap_threshold)
        combined_bearish = combine_overlapping_obs(
            bearish_order_blocks, overlap_threshold)

    # Limit results
    final_bullish = combined_bullish[:bullish_ob_count]
    final_bearish = combined_bearish[:bearish_ob_count]

    # Filter out disabled order blocks
    active_bullish = [ob for ob in final_bullish if not ob.disabled]
    active_bearish = [ob for ob in final_bearish if not ob.disabled]

    print(
        f"Final results: {len(active_bullish)} bullish, {len(active_bearish)} bearish OBs")

    return active_bullish + active_bearish


def evaluate_entry_quality(df: pd.DataFrame, ob: OrderBlockInfo, current_idx: int,
                           all_obs: List[OrderBlockInfo]) -> Tuple[bool, Dict]:
    """Simple entry evaluation - can be enhanced"""
    # Placeholder evaluation - replace with actual logic from flux_orderblock.py
    score = 60  # Default moderate score
    trend_up = True  # Default trend assumption

    # Add some basic trend analysis
    if current_idx >= 20:
        recent_closes = df['close'].iloc[current_idx-20:current_idx]
        trend_up = recent_closes.iloc[-1] > recent_closes.iloc[0]

        if (ob.ob_type == "Bull" and trend_up) or (ob.ob_type == "Bear" and not trend_up):
            score += 20  # Trend confluence

    # Volume analysis
    if ob.ob_volume > df['volume'].iloc[max(0, current_idx-20):current_idx].mean():
        score += 10  # Above average volume

    result = {
        'final_score': score,
        'entry_quality': 'Good' if score >= 70 else 'Moderate' if score >= 50 else 'Poor',
        'risk_level': 'Low' if score >= 70 else 'Medium' if score >= 50 else 'High',
        'warnings': [],
        'confluence': {'primary_trend': 'bullish' if trend_up else 'bearish', 'supporting_timeframes': 1}
    }

    should_take = score >= 45  # Default threshold
    return should_take, result


def combine_overlapping_obs(obs: List[OrderBlockInfo], threshold: float) -> List[OrderBlockInfo]:
    """Combine overlapping order blocks"""
    if not obs or threshold <= 0:
        return obs

    combined = []
    processed = set()

    for i, ob1 in enumerate(obs):
        if i in processed or ob1.disabled:
            continue

        combined_ob = ob1.copy()
        combined_indices = {i}

        # Look for overlapping order blocks of same type
        for j, ob2 in enumerate(obs):
            if j in processed or j == i or ob2.disabled:
                continue
            if ob1.ob_type != ob2.ob_type:
                continue

            # Check overlap (simplified)
            overlap = check_overlap(combined_ob, ob2)
            if overlap > threshold:
                # Combine them
                combined_ob.top = max(combined_ob.top, ob2.top)
                combined_ob.bottom = min(combined_ob.bottom, ob2.bottom)
                combined_ob.ob_volume += ob2.ob_volume
                combined_ob.start_time = min(
                    combined_ob.start_time, ob2.start_time)
                combined_ob.combined = True
                combined_indices.add(j)

        processed.update(combined_indices)
        combined.append(combined_ob)

    return combined


def check_overlap(ob1: OrderBlockInfo, ob2: OrderBlockInfo) -> float:
    """Check overlap percentage between two order blocks"""
    # Simplified overlap calculation - price overlap only
    price_overlap = max(0, min(ob1.top, ob2.top) - max(ob1.bottom, ob2.bottom))
    ob1_height = abs(ob1.top - ob1.bottom)
    ob2_height = abs(ob2.top - ob2.bottom)

    if ob1_height == 0 or ob2_height == 0:
        return 0

    return (price_overlap / min(ob1_height, ob2_height)) * 100


def plot_order_breaker_blocks(df: pd.DataFrame, order_blocks: List[OrderBlockInfo]):
    """Plot order blocks with breaker status"""
    plt.figure(figsize=(15, 10))

    # Plot price data
    plt.plot(df.index, df['close'], label='Close Price',
             color='#2c3e50', linewidth=1)

    # Plot order blocks
    for i, ob in enumerate(order_blocks):
        if ob.disabled:
            continue

        start = ob.start_time
        end = ob.break_time or df.index[-1]

        # Color based on type and breaker status
        if ob.ob_type == "Bull":
            color = '#089981' if not ob.breaker else '#08998150'
            label_prefix = "🟢"
        else:
            color = '#f23646' if not ob.breaker else '#f2364650'
            label_prefix = "🔴"

        alpha = 0.7 if not ob.breaker else 0.3

        # Plot order block rectangle
        plt.fill_betweenx([ob.bottom, ob.top], start, end,
                          color=color, alpha=alpha, edgecolor=color, linewidth=2)

        # Add label with volume info
        mid_price = (ob.top + ob.bottom) / 2
        percentage = int((min(ob.ob_high_volume or 0, ob.ob_low_volume or 0) /
                         max(ob.ob_high_volume or 1, ob.ob_low_volume or 1)) * 100)

        status = "BROKEN" if ob.breaker else "ACTIVE"
        label = f"{label_prefix} {ob.ob_type} {status}\nVol: {ob.ob_volume:.0f} ({percentage}%)"

        if hasattr(ob, 'entry_score') and ob.entry_score:
            label += f"\nScore: {ob.entry_score:.0f}"

        plt.annotate(label, xy=(start, mid_price), xytext=(10, 0),
                     textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    plt.title('Order Blocks & Breaker Blocks Detection',
              fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Order Blocks & Breaker Blocks Detector')
    parser.add_argument('--symbol', type=str,
                        default='SOLUSDT', help='Symbol to analyze')
    parser.add_argument('--interval', type=str,
                        default='15m', help='Timeframe')
    parser.add_argument('--days', type=int, default=7,
                        help='Days of historical data')
    parser.add_argument('--swing_length', type=int,
                        default=10, help='Swing detection length')
    parser.add_argument('--max_atr_mult', type=float,
                        default=3.5, help='Max ATR multiplier')
    parser.add_argument('--ob_end_method', type=str,
                        default='Wick', choices=['Wick', 'Close'])
    parser.add_argument('--bullish_count', type=int,
                        default=3, help='Number of bullish OBs')
    parser.add_argument('--bearish_count', type=int,
                        default=3, help='Number of bearish OBs')
    parser.add_argument('--combine_obs', type=bool,
                        default=True, help='Combine overlapping OBs')
    parser.add_argument('--use_entry_eval', type=bool,
                        default=True, help='Use entry evaluation')

    args = parser.parse_args()

    print(f"🔍 Analyzing {args.symbol} Order Blocks & Breakers...")

    # Fetch data
    client = Client()
    start_time = datetime.now() - timedelta(days=args.days)
    raw_data = client.get_historical_klines(
        args.symbol,
        interval=args.interval,
        start_str=int(start_time.timestamp() * 1000),
        end_str=int(datetime.now().timestamp() * 1000)
    )

    # Process data
    data = pd.DataFrame(raw_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = data[col].astype(float)

    data.set_index('timestamp', inplace=True)

    # Detect order blocks
    order_blocks = detect_order_breaker_blocks(
        data,
        swing_length=args.swing_length,
        max_atr_mult=args.max_atr_mult,
        ob_end_method=args.ob_end_method,
        bullish_ob_count=args.bullish_count,
        bearish_ob_count=args.bearish_count,
        combine_obs=args.combine_obs,
        use_entry_evaluation=args.use_entry_eval
    )

    # Display results
    bullish_obs = [ob for ob in order_blocks if ob.ob_type == "Bull"]
    bearish_obs = [ob for ob in order_blocks if ob.ob_type == "Bear"]

    print(f"\n=== ORDER BLOCKS & BREAKER BLOCKS ANALYSIS ===")
    print(f"Symbol: {args.symbol} | Timeframe: {args.interval}")
    print(
        f"Total Order Blocks: {len(order_blocks)} (Bullish: {len(bullish_obs)}, Bearish: {len(bearish_obs)})")

    print(f"\n--- 🟢 BULLISH ORDER BLOCKS ---")
    for i, ob in enumerate(bullish_obs):
        status = "🔴 BROKEN" if ob.breaker else "🟢 ACTIVE"
        breaker_info = f" | Broken: {ob.break_time}" if ob.breaker else ""
        volume_ratio = f"{ob.ob_low_volume:.0f}/{ob.ob_high_volume:.0f}" if ob.ob_low_volume and ob.ob_high_volume else "N/A"

        info = (f"{i+1}. {status} | Created: {ob.start_time} | "
                f"Range: {ob.bottom:.4f} - {ob.top:.4f} | "
                f"Volume: {ob.ob_volume:.0f} ({volume_ratio}){breaker_info}")

        if hasattr(ob, 'entry_score') and ob.entry_score:
            info += f"\n   📊 Score: {ob.entry_score:.1f} | Quality: {ob.entry_quality} | Risk: {ob.risk_level}"

        print(info)

    print(f"\n--- 🔴 BEARISH ORDER BLOCKS ---")
    for i, ob in enumerate(bearish_obs):
        status = "🔴 BROKEN" if ob.breaker else "🟢 ACTIVE"
        breaker_info = f" | Broken: {ob.break_time}" if ob.breaker else ""
        volume_ratio = f"{ob.ob_low_volume:.0f}/{ob.ob_high_volume:.0f}" if ob.ob_low_volume and ob.ob_high_volume else "N/A"

        info = (f"{i+1}. {status} | Created: {ob.start_time} | "
                f"Range: {ob.bottom:.4f} - {ob.top:.4f} | "
                f"Volume: {ob.ob_volume:.0f} ({volume_ratio}){breaker_info}")

        if hasattr(ob, 'entry_score') and ob.entry_score:
            info += f"\n   📊 Score: {ob.entry_score:.1f} | Quality: {ob.entry_quality} | Risk: {ob.risk_level}"

        print(info)

    # Plot results
    plot_order_breaker_blocks(data, order_blocks)
