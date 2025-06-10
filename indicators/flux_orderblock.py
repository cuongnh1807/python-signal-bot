import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from binance.client import Client

from binance_data_fetcher import BinanceDataFetcher


class SwingPoint:
    """Class to represent swing points"""

    def __init__(self, index=None, price=None, volume=None, crossed=False):
        self.index = index
        self.price = price
        self.volume = volume
        self.crossed = crossed


class OrderBlockInfo:
    """Class to represent order block information"""

    def __init__(self, top, bottom, ob_volume, ob_type, start_time):
        self.top = top
        self.bottom = bottom
        self.ob_volume = ob_volume
        self.ob_type = ob_type  # "Bull" or "Bear"
        self.start_time = start_time
        self.start_index = None
        self.bb_volume = None  # Break volume
        self.ob_low_volume = None
        self.ob_high_volume = None
        self.breaker = False
        self.break_time = None
        self.break_index = None
        self.disabled = False

    def get_area(self, current_time):
        """Calculate area of the order block"""
        end_time = self.break_time if self.break_time else current_time
        width = end_time - self.start_time if end_time > self.start_time else 1
        height = abs(self.top - self.bottom)
        return width * height


def find_swing_points(df, swing_length):
    """
    Find swing highs and lows using the exact Flux Chart Pine Script method

    Parameters:
    - df: DataFrame with OHLCV data
    - swing_length: Length for swing detection

    Returns:
    - swing_highs: List of SwingPoint objects for highs
    - swing_lows: List of SwingPoint objects for lows
    """
    swing_highs = []
    swing_lows = []
    swing_type = 0  # Track current swing state

    # Calculate rolling highest and lowest with lookback
    df['upper'] = df['high'].rolling(window=swing_length).max()
    df['lower'] = df['low'].rolling(window=swing_length).min()

    for i in range(swing_length, len(df)):
        prev_swing_type = swing_type

        # Pine Script logic: swingType := high[len] > upper ? 0 : low[len] < lower ? 1 : swingType
        if df.loc[i-swing_length, 'high'] > df.loc[i, 'upper']:
            swing_type = 0  # High swing
        elif df.loc[i-swing_length, 'low'] < df.loc[i, 'lower']:
            swing_type = 1  # Low swing
        # else keep previous swing_type

        # Create swing points when type changes
        if swing_type == 0 and prev_swing_type != 0:
            swing_highs.append(SwingPoint(
                index=i-swing_length,
                price=df.loc[i-swing_length, 'high'],
                volume=df.loc[i-swing_length, 'volume']
            ))

        if swing_type == 1 and prev_swing_type != 1:
            swing_lows.append(SwingPoint(
                index=i-swing_length,
                price=df.loc[i-swing_length, 'low'],
                volume=df.loc[i-swing_length, 'volume']
            ))

    return swing_highs, swing_lows


def detect_flux_order_blocks(df, swing_length=10, max_atr_mult=3.5, max_order_blocks=30,
                             ob_end_method="Wick", max_distance_to_last_bar=1750,
                             bullish_ob_count=10, bearish_ob_count=10,
                             use_entry_evaluation=True, entry_threshold=45, htf_data=None):
    """
    Detect order blocks using the exact Flux Chart algorithm from Pine Script

    Parameters:
    - use_entry_evaluation: Whether to filter OBs based on entry quality
    - entry_threshold: Minimum entry score required (0-100)
    """
    # Store original index
    original_index = df.index.copy()
    df = df.reset_index(drop=True)

    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(10).mean()
    df['atr'] = df['atr'].fillna(df['tr'].mean())

    # Lists to store order blocks (including broken ones for analysis)
    all_order_blocks = []  # Store all OBs for breaker analysis
    bullish_order_blocks = []
    bearish_order_blocks = []

    # Find swing points using exact Pine Script method
    swing_highs, swing_lows = find_swing_points(df, swing_length)

    print(
        f"Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")

    # Process each bar (Pine Script: if bar_index > last_bar_index - maxDistanceToLastBar)
    for bar_idx in range(len(df)):
        if bar_idx <= len(df) - max_distance_to_last_bar:
            continue

        current_close = df.loc[bar_idx, 'close']
        current_high = df.loc[bar_idx, 'high']
        current_low = df.loc[bar_idx, 'low']
        current_volume = df.loc[bar_idx, 'volume']
        current_atr = df.loc[bar_idx, 'atr']

        # Check mitigation for existing order blocks first
        # Bullish OB mitigation
        for i, ob in enumerate(bullish_order_blocks[:]):
            if not ob.breaker:
                mitigation_price = current_low if ob_end_method == "Wick" else min(
                    df.loc[bar_idx, 'open'], current_close)
                if mitigation_price < ob.bottom:
                    ob.breaker = True
                    ob.break_time = original_index[bar_idx]
                    ob.break_index = bar_idx
                    ob.bb_volume = current_volume
            else:
                # Remove completely broken order blocks
                if current_high > ob.top:
                    bullish_order_blocks.remove(ob)

        # Bearish OB mitigation
        for i, ob in enumerate(bearish_order_blocks[:]):
            if not ob.breaker:
                mitigation_price = current_high if ob_end_method == "Wick" else max(
                    df.loc[bar_idx, 'open'], current_close)
                if mitigation_price > ob.top:
                    ob.breaker = True
                    ob.break_time = original_index[bar_idx]
                    ob.break_index = bar_idx
                    ob.bb_volume = current_volume
            else:
                # Remove completely broken order blocks
                if current_low < ob.bottom:
                    bearish_order_blocks.remove(ob)

        # Check for new order block creation
        # Bullish Order Block creation (Pine Script: if close > top.y and not top.crossed)
        for swing_high in swing_highs:
            if swing_high.index >= bar_idx or swing_high.crossed:
                continue

            if current_close > swing_high.price:
                swing_high.crossed = True
                print(
                    f"Bar {bar_idx}: Bullish trigger - close {current_close:.2f} > swing_high {swing_high.price:.2f}")

                # Pine Script logic for finding order block
                # boxBtm = max[1], boxTop = min[1], boxLoc = time[1]
                if bar_idx >= 1:
                    box_bottom = df.loc[bar_idx-1, 'high']  # max[1]
                    box_top = df.loc[bar_idx-1, 'low']      # min[1]
                    box_location_idx = bar_idx - 1

                    # Pine Script: for i = 1 to (bar_index - top.x) - 1
                    for i in range(1, bar_idx - swing_high.index):
                        if bar_idx - i < 0:
                            break
                        check_idx = bar_idx - i
                        current_min = df.loc[check_idx, 'low']
                        current_max = df.loc[check_idx, 'high']

                        # boxBtm := math.min(min[i], boxBtm)
                        # boxTop := boxBtm == min[i] ? max[i] : boxTop
                        if current_min < box_top:
                            box_top = current_min
                            box_bottom = current_max
                            box_location_idx = check_idx

                    # Calculate volume (Pine Script: volume + volume[1] + volume[2])
                    total_volume = current_volume
                    ob_low_volume = 0
                    ob_high_volume = current_volume

                    if bar_idx >= 1:
                        vol_1 = df.loc[bar_idx-1, 'volume']
                        total_volume += vol_1
                        ob_high_volume += vol_1

                    if bar_idx >= 2:
                        vol_2 = df.loc[bar_idx-2, 'volume']
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
                    ob_info.start_index = box_location_idx
                    ob_info.ob_low_volume = ob_low_volume
                    ob_info.ob_high_volume = ob_high_volume

                    # Size filter
                    ob_size = abs(ob_info.top - ob_info.bottom)
                    if ob_size <= current_atr * max_atr_mult:
                        # Add to all_order_blocks for tracking
                        all_order_blocks.append(ob_info)

                        # Evaluate entry quality
                        if use_entry_evaluation:
                            should_take, eval_result = evaluate_flux_entry(
                                df, ob_info, bar_idx, all_order_blocks, use_entry_evaluation, htf_data
                            )

                            if should_take:
                                # Add evaluation metrics to OB
                                ob_info.entry_score = eval_result['final_score']
                                ob_info.entry_quality = eval_result['entry_quality']
                                ob_info.risk_level = eval_result['risk_level']
                                ob_info.warnings = eval_result['warnings']
                                ob_info.trend_confluence = eval_result['confluence']

                                bullish_order_blocks.insert(0, ob_info)
                                print(
                                    f"✅ Added bullish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}")
                            else:
                                print(
                                    f"❌ Rejected bullish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}")
                        else:
                            bullish_order_blocks.insert(0, ob_info)
                            print(
                                f"Created bullish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}")

                        if len(bullish_order_blocks) > max_order_blocks:
                            bullish_order_blocks.pop()

        # Bearish Order Block creation (Pine Script: if close < btm.y and not btm.crossed)
        for swing_low in swing_lows:
            if swing_low.index >= bar_idx or swing_low.crossed:
                continue

            if current_close < swing_low.price:
                swing_low.crossed = True
                print(
                    f"Bar {bar_idx}: Bearish trigger - close {current_close:.2f} < swing_low {swing_low.price:.2f}")

                # Pine Script logic for bearish order block
                if bar_idx >= 1:
                    box_bottom = df.loc[bar_idx-1, 'low']   # min[1]
                    box_top = df.loc[bar_idx-1, 'high']    # max[1]
                    box_location_idx = bar_idx - 1

                    # Pine Script: for i = 1 to (bar_index - btm.x) - 1
                    for i in range(1, bar_idx - swing_low.index):
                        if bar_idx - i < 0:
                            break
                        check_idx = bar_idx - i
                        current_max = df.loc[check_idx, 'high']
                        current_min = df.loc[check_idx, 'low']

                        # boxTop := math.max(max[i], boxTop)
                        # boxBtm := boxTop == max[i] ? min[i] : boxBtm
                        if current_max > box_top:
                            box_top = current_max
                            box_bottom = current_min
                            box_location_idx = check_idx

                    # Calculate volume for bearish
                    total_volume = current_volume
                    ob_low_volume = current_volume
                    ob_high_volume = 0

                    if bar_idx >= 1:
                        vol_1 = df.loc[bar_idx-1, 'volume']
                        total_volume += vol_1
                        ob_low_volume += vol_1

                    if bar_idx >= 2:
                        vol_2 = df.loc[bar_idx-2, 'volume']
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
                    ob_info.start_index = box_location_idx
                    ob_info.ob_low_volume = ob_low_volume
                    ob_info.ob_high_volume = ob_high_volume

                    # Size filter
                    ob_size = abs(ob_info.top - ob_info.bottom)
                    if ob_size <= current_atr * max_atr_mult:
                        # Add to all_order_blocks for tracking
                        all_order_blocks.append(ob_info)

                        # Evaluate entry quality
                        if use_entry_evaluation:
                            should_take, eval_result = evaluate_flux_entry(
                                df, ob_info, bar_idx, all_order_blocks, use_entry_evaluation
                            )

                            if should_take:
                                # Add evaluation metrics to OB
                                ob_info.entry_score = eval_result['final_score']
                                ob_info.entry_quality = eval_result['entry_quality']
                                ob_info.risk_level = eval_result['risk_level']
                                ob_info.warnings = eval_result['warnings']
                                ob_info.trend_confluence = eval_result['confluence']

                                bearish_order_blocks.insert(0, ob_info)
                                print(
                                    f"✅ Added bearish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}")
                            else:
                                print(
                                    f"❌ Rejected bearish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}")
                        else:
                            bearish_order_blocks.insert(0, ob_info)
                            print(
                                f"Created bearish OB: {ob_info.bottom:.4f} - {ob_info.top:.4f}")

                        if len(bearish_order_blocks) > max_order_blocks:
                            bearish_order_blocks.pop()

    # Remove duplicates and sort by recency
    def remove_duplicates(obs_list):
        """Remove duplicate order blocks with same price range"""
        unique_obs = []
        seen_ranges = set()

        for ob in obs_list:
            range_key = (round(ob.top, 2), round(ob.bottom, 2))
            if range_key not in seen_ranges:
                seen_ranges.add(range_key)
                unique_obs.append(ob)
        return unique_obs

    # Remove duplicates and sort by recency
    bullish_order_blocks = remove_duplicates(bullish_order_blocks)
    bearish_order_blocks = remove_duplicates(bearish_order_blocks)

    # Sort by entry score if evaluation is used, otherwise by start time
    if use_entry_evaluation:
        bullish_order_blocks.sort(key=lambda x: getattr(
            x, 'entry_score', 0), reverse=True)
        bearish_order_blocks.sort(key=lambda x: getattr(
            x, 'entry_score', 0), reverse=True)
    else:
        bullish_order_blocks.sort(key=lambda x: x.start_time, reverse=True)
        bearish_order_blocks.sort(key=lambda x: x.start_time, reverse=True)

    bullish_obs = bullish_order_blocks[:bullish_ob_count]
    bearish_obs = bearish_order_blocks[:bearish_ob_count]

    # Filter out disabled
    active_bullish = [ob for ob in bullish_obs if not ob.disabled]
    active_bearish = [ob for ob in bearish_obs if not ob.disabled]

    return active_bullish + active_bearish


def calculate_overlap_percentage(ob1, ob2, current_time):
    """Calculate overlap percentage between two order blocks"""
    # Time boundaries
    xa1 = ob1.start_time
    xa2 = ob1.break_time if ob1.break_time else current_time
    ya1 = ob1.top
    ya2 = ob1.bottom

    xb1 = ob2.start_time
    xb2 = ob2.break_time if ob2.break_time else current_time
    yb1 = ob2.top
    yb2 = ob2.bottom

    # Calculate intersection area
    x_overlap = max(0, min(xa2, xb2) - max(xa1, xb1))
    y_overlap = max(0, min(ya1, yb1) - max(ya2, yb2))
    intersection_area = x_overlap * y_overlap

    # Calculate union area
    area1 = ob1.get_area(current_time)
    area2 = ob2.get_area(current_time)
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0

    return (intersection_area / union_area) * 100.0


def combine_overlapping_order_blocks(order_blocks, overlap_threshold=0, current_time=None):
    """Combine overlapping order blocks"""
    if not order_blocks or overlap_threshold <= 0:
        return order_blocks

    if current_time is None:
        current_time = datetime.now()

    combined_blocks = []
    processed = set()

    for i, ob1 in enumerate(order_blocks):
        if i in processed or ob1.disabled:
            continue

        # Start with current order block
        combined_ob = ob1
        combined_indices = {i}

        # Look for overlapping order blocks of same type
        for j, ob2 in enumerate(order_blocks):
            if j in processed or j == i or ob2.disabled:
                continue

            if ob1.ob_type != ob2.ob_type:
                continue

            overlap_pct = calculate_overlap_percentage(
                combined_ob, ob2, current_time)
            if overlap_pct > overlap_threshold:
                # Combine the order blocks
                new_top = max(combined_ob.top, ob2.top)
                new_bottom = min(combined_ob.bottom, ob2.bottom)
                new_volume = combined_ob.ob_volume + ob2.ob_volume
                new_start_time = min(combined_ob.start_time, ob2.start_time)

                # Create new combined order block
                combined_ob = OrderBlockInfo(
                    top=new_top,
                    bottom=new_bottom,
                    ob_volume=new_volume,
                    ob_type=ob1.ob_type,
                    start_time=new_start_time
                )

                # Combine other properties
                combined_ob.ob_low_volume = (
                    combined_ob.ob_low_volume or 0) + (ob2.ob_low_volume or 0)
                combined_ob.ob_high_volume = (
                    combined_ob.ob_high_volume or 0) + (ob2.ob_high_volume or 0)
                combined_ob.breaker = combined_ob.breaker or ob2.breaker

                if combined_ob.break_time and ob2.break_time:
                    combined_ob.break_time = max(
                        combined_ob.break_time, ob2.break_time)
                elif ob2.break_time:
                    combined_ob.break_time = ob2.break_time

                combined_indices.add(j)

        # Mark all combined indices as processed
        processed.update(combined_indices)
        combined_blocks.append(combined_ob)

    return combined_blocks


def plot_flux_order_blocks(df, order_blocks):
    """
    Visualize Flux Chart order blocks on price chart

    Parameters:
        df (pd.DataFrame): OHLCV DataFrame
        order_blocks (list): List of OrderBlockInfo objects
    """
    plt.figure(figsize=(15, 8))

    # Plot price data
    plt.plot(df.index, df['close'], label='Close Price',
             color='#2c3e50', linewidth=1)

    # Plot order blocks
    for ob in order_blocks:
        start = ob.start_time
        end = ob.break_time or df.index[-1]

        if ob.ob_type == "Bull":
            color = '#089981'
            alpha = 0.3 if not ob.breaker else 0.15
        else:
            color = '#f23646'
            alpha = 0.3 if not ob.breaker else 0.15

        # Plot order block rectangle
        plt.fill_betweenx([ob.bottom, ob.top], start, end,
                          color=color, alpha=alpha, edgecolor=color)

        # Plot center line
        plt.hlines(ob.bottom + (ob.top - ob.bottom)/2, start, end,
                   colors=color, linestyles='dashed', linewidth=1, alpha=0.7)

    # Format chart
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.title('Flux Chart Order Block Detection',
              fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_trend_confluence(df, current_idx, timeframe_multipliers=[3, 6, 12]):
    """
    Analyze trend on multiple timeframes for confluence

    Parameters:
    - df: DataFrame with OHLCV data
    - current_idx: Current bar index
    - timeframe_multipliers: Multipliers for higher timeframes

    Returns:
    - dict: Trend analysis results
    """
    trend_analysis = {
        'primary_trend': 'neutral',
        'trend_strength': 0,
        'confluence_score': 0,
        'trend_change_detected': False,
        'supporting_timeframes': 0
    }

    if current_idx < 50:
        return trend_analysis

    # Calculate EMAs for trend analysis
    ema_periods = [21, 50, 100]
    trend_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}

    for period in ema_periods:
        if current_idx >= period:
            ema = df['close'].rolling(window=period).mean()
            current_price = df['close'].iloc[current_idx]
            ema_current = ema.iloc[current_idx]
            ema_prev = ema.iloc[current_idx -
                                5] if current_idx >= 5 else ema_current

            # Trend direction based on price vs EMA and EMA slope
            if current_price > ema_current and ema_current > ema_prev:
                trend_votes['bullish'] += 1
            elif current_price < ema_current and ema_current < ema_prev:
                trend_votes['bearish'] += 1
            else:
                trend_votes['neutral'] += 1

    # Determine primary trend
    max_votes = max(trend_votes.values())
    if trend_votes['bullish'] == max_votes and trend_votes['bullish'] >= 2:
        trend_analysis['primary_trend'] = 'bullish'
        trend_analysis['trend_strength'] = (trend_votes['bullish'] / 3) * 100
    elif trend_votes['bearish'] == max_votes and trend_votes['bearish'] >= 2:
        trend_analysis['primary_trend'] = 'bearish'
        trend_analysis['trend_strength'] = (trend_votes['bearish'] / 3) * 100

    # Supporting timeframes (simulate higher TF analysis)
    supporting_count = 0
    for mult in timeframe_multipliers:
        if current_idx >= mult * 21:
            htf_ema = df['close'].rolling(window=mult * 21).mean()
            htf_current = htf_ema.iloc[current_idx]
            htf_prev = htf_ema.iloc[current_idx -
                                    mult] if current_idx >= mult else htf_current

            if trend_analysis['primary_trend'] == 'bullish' and htf_current > htf_prev:
                supporting_count += 1
            elif trend_analysis['primary_trend'] == 'bearish' and htf_current < htf_prev:
                supporting_count += 1

    trend_analysis['supporting_timeframes'] = supporting_count
    trend_analysis['confluence_score'] = (
        supporting_count / len(timeframe_multipliers)) * 100

    # Detect potential trend changes
    if current_idx >= 10:
        recent_highs = df['high'].iloc[current_idx-10:current_idx+1].max()
        recent_lows = df['low'].iloc[current_idx-10:current_idx+1].min()
        current_close = df['close'].iloc[current_idx]

        if (trend_analysis['primary_trend'] == 'bearish' and
                current_close > (recent_lows + (recent_highs - recent_lows) * 0.7)):
            trend_analysis['trend_change_detected'] = True
        elif (trend_analysis['primary_trend'] == 'bullish' and
              current_close < (recent_lows + (recent_highs - recent_lows) * 0.3)):
            trend_analysis['trend_change_detected'] = True

    return trend_analysis


def evaluate_breaker_obs(order_blocks, current_ob, current_idx):
    """
    Evaluate recently broken order blocks for confluence/divergence

    Parameters:
    - order_blocks: List of all order blocks (including broken ones)
    - current_ob: Current order block to evaluate
    - current_idx: Current bar index

    Returns:
    - dict: Breaker analysis results
    """
    breaker_analysis = {
        'recent_breakers': 0,
        'opposing_breakers': 0,
        'confluence_breakers': 0,
        'breaker_strength': 0
    }

    # Look for recently broken OBs (within last 50 bars)
    recent_breakers = [
        ob for ob in order_blocks
        if (hasattr(ob, 'breaker') and ob.breaker and
            hasattr(ob, 'break_index') and
            ob.break_index is not None and
            current_idx - ob.break_index <= 50)
    ]

    breaker_analysis['recent_breakers'] = len(recent_breakers)

    # Analyze breaker directions vs current OB
    for breaker_ob in recent_breakers:
        if breaker_ob.ob_type != current_ob.ob_type:
            breaker_analysis['opposing_breakers'] += 1
        else:
            breaker_analysis['confluence_breakers'] += 1

    # Calculate breaker strength
    if recent_breakers:
        total_volume = sum(getattr(ob, 'bb_volume', 0)
                           for ob in recent_breakers)
        breaker_analysis['breaker_strength'] = min(
            total_volume / len(recent_breakers), 100)

    return breaker_analysis


def evaluate_flux_entry(df, order_block, current_idx, all_order_blocks=None, use_evaluation=True, htf_data=None):
    """
    Enhanced Flux Chart order block entry quality evaluation with 4h timeframe analysis

    Parameters:
    - df: DataFrame with OHLCV data
    - order_block: OrderBlockInfo object to evaluate
    - current_idx: Current bar index
    - all_order_blocks: List of all order blocks for breaker analysis
    - use_evaluation: If False, always return positive evaluation
    - htf_data: Higher timeframe (4h) data for confluence analysis

    Returns:
    - tuple: (should_take_entry: bool, analysis_result: dict)
    """
    if not use_evaluation:
        return True, {
            "setup_quality": 100,
            "final_score": 100,
            "threshold": 0,
            "warnings": [],
            "entry_quality": "Excellent"
        }

    ob_direction = 1 if order_block.ob_type == "Bull" else -1
    ob_direction_str = order_block.ob_type.upper()
    warnings = []

    # 1. Enhanced Trend Analysis with 4h Confluence (40% weight)
    trend_analysis = analyze_enhanced_trend_confluence(
        df, current_idx, htf_data)
    trend_score = 0

    # Primary timeframe trend assessment
    primary_trend_bonus = 0
    if trend_analysis['primary_trend'] == 'bullish' and ob_direction == 1:
        # Bullish OB in bullish trend
        primary_trend_bonus = 60
    elif trend_analysis['primary_trend'] == 'bearish' and ob_direction == -1:
        # Bearish OB in bearish trend
        primary_trend_bonus = 60
    elif trend_analysis['primary_trend'] == 'bullish' and ob_direction == -1:
        # Counter-trend bearish OB - potential reversal
        if trend_analysis['trend_change_detected']:
            primary_trend_bonus = 70  # Strong reversal signal
        else:
            primary_trend_bonus = 30  # Risky counter-trend
            warnings.append("Counter-trend bearish OB in bullish market")
    elif trend_analysis['primary_trend'] == 'bearish' and ob_direction == 1:
        # Counter-trend bullish OB - potential reversal
        if trend_analysis['trend_change_detected']:
            primary_trend_bonus = 70  # Strong reversal signal
        else:
            primary_trend_bonus = 30  # Risky counter-trend
            warnings.append("Counter-trend bullish OB in bearish market")
    else:
        # Neutral trend
        primary_trend_bonus = 50

    # Higher timeframe confluence bonus
    htf_confluence_bonus = 0
    if htf_data is not None and len(htf_data) > 0:
        htf_trend = trend_analysis.get('htf_trend', 'neutral')
        htf_strength = trend_analysis.get('htf_strength', 0)

        if htf_trend == 'bullish' and ob_direction == 1:
            htf_confluence_bonus = 25 + \
                (htf_strength * 0.15)  # Up to 37 points
        elif htf_trend == 'bearish' and ob_direction == -1:
            htf_confluence_bonus = 25 + \
                (htf_strength * 0.15)  # Up to 37 points
        elif htf_trend != 'neutral' and ((htf_trend == 'bullish' and ob_direction == -1) or (htf_trend == 'bearish' and ob_direction == 1)):
            # HTF opposes OB direction
            htf_confluence_bonus = -15
            warnings.append(f"4h trend ({htf_trend}) opposes OB direction")
        else:
            htf_confluence_bonus = 5  # Neutral HTF

        # Additional bonus for strong multi-timeframe confluence
        if trend_analysis.get('supporting_timeframes', 0) >= 2:
            htf_confluence_bonus += 10
    else:
        # No HTF data available
        htf_confluence_bonus = 0
        warnings.append("No 4h data available for confluence")

    trend_score = min(primary_trend_bonus + htf_confluence_bonus, 100)

    # 2. Market Structure Score (25% weight)
    structure_score = 50  # Default neutral

    # Enhanced market structure: higher highs/lows for bullish, lower highs/lows for bearish
    if current_idx >= 20:
        recent_data = df.iloc[current_idx-20:current_idx+1]
        highs = recent_data['high']
        lows = recent_data['low']

        recent_high = highs.max()
        recent_low = lows.min()
        prev_high = df['high'].iloc[max(
            0, current_idx-40):current_idx-20].max()
        prev_low = df['low'].iloc[max(0, current_idx-40):current_idx-20].min()

        if ob_direction == 1:  # Bullish OB
            if recent_high > prev_high and recent_low > prev_low:
                structure_score = 85  # Higher highs and higher lows
            elif recent_high > prev_high:
                structure_score = 70  # Just higher highs
            elif recent_low < prev_low:
                structure_score = 30  # Lower lows - concerning for bullish
                warnings.append("Market making lower lows")
            else:
                structure_score = 55  # Neutral structure
        else:  # Bearish OB
            if recent_high < prev_high and recent_low < prev_low:
                structure_score = 85  # Lower highs and lower lows
            elif recent_low < prev_low:
                structure_score = 70  # Just lower lows
            elif recent_high > prev_high:
                structure_score = 30  # Higher highs - concerning for bearish
                warnings.append("Market making higher highs")
            else:
                structure_score = 55  # Neutral structure

    # 3. Enhanced Breaker Analysis (20% weight)
    breaker_score = 50  # Default neutral
    if all_order_blocks:
        breaker_analysis = evaluate_breaker_obs(
            all_order_blocks, order_block, current_idx)

        if breaker_analysis['recent_breakers'] > 0:
            # Penalize if recent breakers oppose current OB direction
            opposing_ratio = breaker_analysis['opposing_breakers'] / \
                breaker_analysis['recent_breakers']
            confluence_ratio = breaker_analysis['confluence_breakers'] / \
                breaker_analysis['recent_breakers']

            if confluence_ratio > 0.7:
                breaker_score = 80  # Most recent breakers support current direction
            elif confluence_ratio > 0.5:
                breaker_score = 65  # Moderate support
            elif opposing_ratio > 0.7:
                breaker_score = 25  # Most recent breakers oppose current direction
                warnings.append(
                    f"{breaker_analysis['opposing_breakers']} recent opposing breakers")
            elif opposing_ratio > 0.5:
                breaker_score = 40  # Some opposition
            else:
                breaker_score = 55  # Mixed signals

            # Boost score if breaker strength is high
            if breaker_analysis['breaker_strength'] > 50:
                breaker_score += 10

    # 4. Enhanced Volume & Momentum (15% weight)
    volume_score = 50
    if hasattr(order_block, 'ob_volume') and order_block.ob_volume > 0:
        # Use relative volume strength
        if current_idx >= 20:
            avg_volume = df['volume'].iloc[current_idx-20:current_idx].mean()
            volume_ratio = order_block.ob_volume / avg_volume if avg_volume > 0 else 1

            if volume_ratio >= 2.0:
                volume_score = 90  # Exceptional volume
            elif volume_ratio >= 1.5:
                volume_score = 75  # High volume
            elif volume_ratio >= 1.2:
                volume_score = 60  # Above average
            elif volume_ratio >= 0.8:
                volume_score = 45  # Below average
            else:
                volume_score = 30  # Low volume
                warnings.append("Below average volume")

        # Additional momentum check using recent price action
        if current_idx >= 5:
            recent_momentum = df['close'].iloc[current_idx] - \
                df['close'].iloc[current_idx-5]
            if (ob_direction == 1 and recent_momentum > 0) or (ob_direction == -1 and recent_momentum < 0):
                volume_score += 10  # Momentum supports direction

    # Calculate weighted final score
    weights = {
        'trend': 0.40,
        'structure': 0.25,
        'breaker': 0.20,
        'volume': 0.15
    }

    final_score = (
        trend_score * weights['trend'] +
        structure_score * weights['structure'] +
        breaker_score * weights['breaker'] +
        volume_score * weights['volume']
    )

    # Enhanced Adaptive threshold based on market conditions
    base_threshold = 45

    # Increase threshold for counter-trend setups
    if ((trend_analysis['primary_trend'] == 'bullish' and ob_direction == -1) or
            (trend_analysis['primary_trend'] == 'bearish' and ob_direction == 1)):
        if not trend_analysis['trend_change_detected']:
            base_threshold = 65  # Higher bar for counter-trend

    # Increase threshold if HTF opposes
    if htf_data is not None and trend_analysis.get('htf_trend', 'neutral') != 'neutral':
        htf_trend = trend_analysis['htf_trend']
        if ((htf_trend == 'bullish' and ob_direction == -1) or
                (htf_trend == 'bearish' and ob_direction == 1)):
            base_threshold += 20  # Much higher bar for HTF opposing setups

    # Increase threshold if many recent opposing breakers
    if all_order_blocks:
        breaker_analysis = evaluate_breaker_obs(
            all_order_blocks, order_block, current_idx)
        if (breaker_analysis['recent_breakers'] > 0 and
                breaker_analysis['opposing_breakers'] / breaker_analysis['recent_breakers'] > 0.6):
            base_threshold += 15

    threshold = min(base_threshold, 80)

    # Determine entry quality based on enhanced scoring
    setup_quality = final_score
    if setup_quality >= 85:
        entry_quality = "Excellent"
        risk_level = "Low"
    elif setup_quality >= 70:
        entry_quality = "Good"
        risk_level = "Low-Medium"
    elif setup_quality >= 55:
        entry_quality = "Moderate"
        risk_level = "Medium"
    elif setup_quality >= 40:
        entry_quality = "Poor"
        risk_level = "High"
    else:
        entry_quality = "Very Poor"
        risk_level = "Very High"

    should_take_entry = final_score >= threshold

    # Enhanced debug output
    if should_take_entry:
        print(f"✅ EXCELLENT FLUX ENTRY {ob_direction_str}: Score {final_score:.1f} | "
              f"Trend: {trend_score:.1f}, Structure: {structure_score:.1f}, "
              f"Breaker: {breaker_score:.1f}, Volume: {volume_score:.1f}")
        if htf_data is not None:
            htf_trend = trend_analysis.get('htf_trend', 'neutral')
            print(
                f"   📈 4h Trend: {htf_trend.title()} (Strength: {trend_analysis.get('htf_strength', 0):.0f})")
        if trend_analysis['trend_change_detected']:
            print(f"   🔄 Trend change detected - potential reversal setup")
        if trend_analysis['supporting_timeframes'] >= 2:
            print(
                f"   📊 {trend_analysis['supporting_timeframes']} supporting timeframes")
    else:
        print(
            f"❌ REJECTED FLUX ENTRY {ob_direction_str}: Score {final_score:.1f} < {threshold:.1f}")
        if htf_data is not None:
            htf_trend = trend_analysis.get('htf_trend', 'neutral')
            print(f"   📈 4h Trend: {htf_trend.title()} (conflicts with setup)" if htf_trend !=
                  'neutral' else f"   📈 4h Trend: Neutral")
        for warning in warnings:
            print(f"   ⚠️  {warning}")

    # Enhanced result compilation
    result = {
        "setup_quality": setup_quality,
        "final_score": final_score,
        "threshold": threshold,
        "entry_quality": entry_quality,
        "risk_level": risk_level,
        "warnings": warnings,
        "scores": {
            "trend": trend_score,
            "structure": structure_score,
            "breaker": breaker_score,
            "volume": volume_score
        },
        "trend_analysis": trend_analysis,
        "confluence": {
            "primary_trend": trend_analysis['primary_trend'],
            "htf_trend": trend_analysis.get('htf_trend', 'neutral'),
            "htf_strength": trend_analysis.get('htf_strength', 0),
            "supporting_timeframes": trend_analysis['supporting_timeframes'],
            "trend_change": trend_analysis['trend_change_detected'],
            "confluence_score": trend_analysis.get('confluence_score', 50)
        }
    }

    return should_take_entry, result


def analyze_enhanced_trend_confluence(df: pd.DataFrame, current_idx: int, htf_data: pd.DataFrame = None):
    """
    Enhanced trend analysis with 4h timeframe support for Flux OrderBlocks

    Parameters:
    - df: Primary timeframe data
    - current_idx: Current bar index
    - htf_data: 4h timeframe data for confluence

    Returns:
    - dict: Enhanced trend analysis results
    """
    trend_analysis = {
        'primary_trend': 'neutral',
        'trend_strength': 0,
        'confluence_score': 50,
        'trend_change_detected': False,
        'supporting_timeframes': 0,
        'htf_trend': 'neutral',
        'htf_strength': 0
    }

    if current_idx < 50:
        return trend_analysis

    # Calculate EMAs for trend analysis on primary timeframe
    ema_periods = [21, 50, 100]
    trend_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}

    for period in ema_periods:
        if current_idx >= period:
            ema = df['close'].rolling(window=period).mean()
            current_price = df['close'].iloc[current_idx]
            ema_current = ema.iloc[current_idx]
            ema_prev = ema.iloc[current_idx -
                                5] if current_idx >= 5 else ema_current

            # Trend direction based on price vs EMA and EMA slope
            if current_price > ema_current and ema_current > ema_prev:
                trend_votes['bullish'] += 1
            elif current_price < ema_current and ema_current < ema_prev:
                trend_votes['bearish'] += 1
            else:
                trend_votes['neutral'] += 1

    # Determine primary trend
    max_votes = max(trend_votes.values())
    if trend_votes['bullish'] == max_votes and trend_votes['bullish'] >= 2:
        trend_analysis['primary_trend'] = 'bullish'
        trend_analysis['trend_strength'] = (trend_votes['bullish'] / 3) * 100
    elif trend_votes['bearish'] == max_votes and trend_votes['bearish'] >= 2:
        trend_analysis['primary_trend'] = 'bearish'
        trend_analysis['trend_strength'] = (trend_votes['bearish'] / 3) * 100

    # Enhanced 4h timeframe analysis
    if htf_data is not None and len(htf_data) > 21:
        try:
            # Calculate 4h EMAs
            htf_ema_21 = htf_data['close'].rolling(window=21).mean()
            htf_ema_50 = htf_data['close'].rolling(window=50).mean()
            htf_ema_100 = htf_data['close'].rolling(
                window=100).mean() if len(htf_data) > 100 else htf_ema_50

            if len(htf_ema_21) > 1 and len(htf_ema_50) > 1:
                htf_current_price = htf_data['close'].iloc[-1]
                htf_ema21_current = htf_ema_21.iloc[-1]
                htf_ema50_current = htf_ema_50.iloc[-1]
                htf_ema100_current = htf_ema_100.iloc[-1]

                # Enhanced HTF trend determination with strength calculation
                htf_score = 0

                # Price position relative to EMAs
                if htf_current_price > htf_ema21_current:
                    htf_score += 30
                if htf_current_price > htf_ema50_current:
                    htf_score += 25
                if len(htf_data) > 100 and htf_current_price > htf_ema100_current:
                    htf_score += 25

                # EMA alignment
                if htf_ema21_current > htf_ema50_current:
                    htf_score += 10
                if len(htf_data) > 100 and htf_ema50_current > htf_ema100_current:
                    htf_score += 10

                # Determine HTF trend and strength
                if htf_score >= 70:
                    trend_analysis['htf_trend'] = 'bullish'
                    trend_analysis['htf_strength'] = min(htf_score, 100)
                elif htf_score <= 30:
                    trend_analysis['htf_trend'] = 'bearish'
                    trend_analysis['htf_strength'] = min(100 - htf_score, 100)
                else:
                    trend_analysis['htf_trend'] = 'neutral'
                    trend_analysis['htf_strength'] = 50

                # Enhanced confluence calculation
                supporting_count = 0
                if trend_analysis['primary_trend'] == trend_analysis['htf_trend'] and trend_analysis['htf_trend'] != 'neutral':
                    # Strong confluence
                    supporting_count = 3
                    confluence_bonus = trend_analysis['htf_strength'] * 0.4
                    trend_analysis['confluence_score'] = 70 + confluence_bonus
                elif trend_analysis['primary_trend'] != 'neutral' and trend_analysis['htf_trend'] != 'neutral':
                    # Conflicting trends
                    supporting_count = 0
                    trend_analysis['confluence_score'] = 25
                else:
                    # One neutral
                    supporting_count = 1
                    trend_analysis['confluence_score'] = 50

                trend_analysis['supporting_timeframes'] = supporting_count

        except Exception as e:
            print(f"Warning: HTF analysis failed: {e}")
            # Fallback to primary timeframe only
            trend_analysis['htf_trend'] = 'neutral'
            trend_analysis['htf_strength'] = 0

    # Enhanced trend change detection with HTF consideration
    if current_idx >= 10:
        recent_highs = df['high'].iloc[current_idx-10:current_idx+1].max()
        recent_lows = df['low'].iloc[current_idx-10:current_idx+1].min()
        current_close = df['close'].iloc[current_idx]

        # Check for potential reversal patterns
        reversal_threshold = 0.7 if trend_analysis['primary_trend'] == 'bearish' else 0.3

        if (trend_analysis['primary_trend'] == 'bearish' and
                current_close > (recent_lows + (recent_highs - recent_lows) * reversal_threshold)):
            trend_analysis['trend_change_detected'] = True
        elif (trend_analysis['primary_trend'] == 'bullish' and
              current_close < (recent_lows + (recent_highs - recent_lows) * reversal_threshold)):
            trend_analysis['trend_change_detected'] = True

        # HTF support for trend change
        if (trend_analysis['trend_change_detected'] and htf_data is not None and
                trend_analysis['htf_trend'] != 'neutral'):
            # If HTF supports the potential reversal, boost confidence
            primary_reversing_to = 'bullish' if trend_analysis[
                'primary_trend'] == 'bearish' else 'bearish'
            if trend_analysis['htf_trend'] == primary_reversing_to:
                # HTF supports reversal
                trend_analysis['confluence_score'] += 15

    return trend_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Flux Chart Order Block Detector with Entry Evaluation')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Symbol to fetch data')
    parser.add_argument('--interval', type=str, default='15m',
                        help='Interval to fetch data')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days of historical data')
    parser.add_argument('--swing_length', type=int, default=10,
                        help='Swing length for detection')
    parser.add_argument('--max_atr_mult', type=float, default=3.5,
                        help='Maximum ATR multiplier for size filtering')
    parser.add_argument('--mitigation', type=str, default='Wick',
                        choices=['Wick', 'Close'], help='Mitigation method')
    parser.add_argument('--bullish_count', type=int, default=3,
                        help='Number of bullish OBs to display')
    parser.add_argument('--bearish_count', type=int, default=3,
                        help='Number of bearish OBs to display')
    parser.add_argument('--use_entry_eval', type=str, default='true',
                        choices=['true', 'false'], help='Use entry evaluation filtering')
    parser.add_argument('--entry_threshold', type=int, default=45,
                        help='Minimum entry score required (0-100)')

    args = parser.parse_args()

    # Fetch data
    client = Client()
    start_time = datetime.now() - timedelta(days=args.days)
    raw_data = client.get_historical_klines(
        args.symbol,
        interval=args.interval,
        start_str=int(start_time.timestamp() * 1000),
        end_str=int((datetime.now()).timestamp() * 1000)
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

    # Convert string to boolean
    use_entry_eval = args.use_entry_eval.lower() == 'true'

    # Detect order blocks
    order_blocks = detect_flux_order_blocks(
        data,
        swing_length=args.swing_length,
        max_atr_mult=args.max_atr_mult,
        ob_end_method=args.mitigation,
        bullish_ob_count=args.bullish_count,
        bearish_ob_count=args.bearish_count,
        use_entry_evaluation=use_entry_eval,
        entry_threshold=args.entry_threshold
    )

    # Combine overlapping order blocks
    order_blocks = combine_overlapping_order_blocks(
        order_blocks, overlap_threshold=0)

    # Separate and display results
    bullish_obs = [ob for ob in order_blocks if ob.ob_type == "Bull"]
    bearish_obs = [ob for ob in order_blocks if ob.ob_type == "Bear"]

    print(f"\n=== FLUX CHART ORDER BLOCKS WITH ENTRY EVALUATION ===")
    print(
        f"Symbol: {args.symbol} | Timeframe: {args.interval} | Swing Length: {args.swing_length}")
    print(f"Entry Evaluation: {'✅ ENABLED' if use_entry_eval else '❌ DISABLED'} | "
          f"Threshold: {args.entry_threshold if use_entry_eval else 'N/A'}")
    print(
        f"Total Quality Order Blocks: {len(order_blocks)} (Bullish: {len(bullish_obs)}, Bearish: {len(bearish_obs)})")

    print(f"\n--- 🟢 BULLISH ORDER BLOCKS ---")
    for i, ob in enumerate(bullish_obs):
        status = "🔴 BROKEN" if ob.breaker else "🟢 ACTIVE"
        volume_ratio = f"{ob.ob_low_volume:.0f}/{ob.ob_high_volume:.0f}" if ob.ob_low_volume and ob.ob_high_volume else "N/A"

        basic_info = (f"{i+1}. {status} | Time: {ob.start_time} | "
                      f"Range: {ob.bottom:.4f} - {ob.top:.4f} | "
                      f"Volume: {ob.ob_volume:.0f} ({volume_ratio}) | "
                      f"Height: {abs(ob.top-ob.bottom):.6f}")

        if use_entry_eval and hasattr(ob, 'entry_score'):
            quality_info = (f"\n   📊 Entry Score: {ob.entry_score:.1f} | "
                            f"Quality: {ob.entry_quality} | Risk: {ob.risk_level}")

            confluence_info = ""
            if hasattr(ob, 'trend_confluence'):
                conf = ob.trend_confluence
                confluence_info = (f"\n   🔄 Trend: {conf['primary_trend'].title()} | "
                                   f"Supporting TFs: {conf['supporting_timeframes']}/3")
                if conf['trend_change']:
                    confluence_info += " | 🔄 Trend Change Detected"

            warnings_info = ""
            if hasattr(ob, 'warnings') and ob.warnings:
                warnings_info = f"\n   ⚠️  Warnings: {', '.join(ob.warnings)}"

            print(basic_info + quality_info + confluence_info + warnings_info)
        else:
            print(basic_info)

    print(f"\n--- 🔴 BEARISH ORDER BLOCKS ---")
    for i, ob in enumerate(bearish_obs):
        status = "🔴 BROKEN" if ob.breaker else "🟢 ACTIVE"
        volume_ratio = f"{ob.ob_low_volume:.0f}/{ob.ob_high_volume:.0f}" if ob.ob_low_volume and ob.ob_high_volume else "N/A"

        basic_info = (f"{i+1}. {status} | Time: {ob.start_time} | "
                      f"Range: {ob.bottom:.4f} - {ob.top:.4f} | "
                      f"Volume: {ob.ob_volume:.0f} ({volume_ratio}) | "
                      f"Height: {abs(ob.top-ob.bottom):.6f}")

        if use_entry_eval and hasattr(ob, 'entry_score'):
            quality_info = (f"\n   📊 Entry Score: {ob.entry_score:.1f} | "
                            f"Quality: {ob.entry_quality} | Risk: {ob.risk_level}")

            confluence_info = ""
            if hasattr(ob, 'trend_confluence'):
                conf = ob.trend_confluence
                confluence_info = (f"\n   🔄 Trend: {conf['primary_trend'].title()} | "
                                   f"Supporting TFs: {conf['supporting_timeframes']}/3")
                if conf['trend_change']:
                    confluence_info += " | 🔄 Trend Change Detected"

            warnings_info = ""
            if hasattr(ob, 'warnings') and ob.warnings:
                warnings_info = f"\n   ⚠️  Warnings: {', '.join(ob.warnings)}"

            print(basic_info + quality_info + confluence_info + warnings_info)
        else:
            print(basic_info)

    # Plot results
    plot_flux_order_blocks(data, order_blocks)
