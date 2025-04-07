from typing import Tuple, Dict
import pandas as pd
from indicators.rsi import calculate_rsi, calculate_macd


def is_pin_bar(row):
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - max(row['open'], row['close'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    if body < 0.1 * (row['high'] - row['low']):
        if upper_shadow > 2 * body:
            return -1
        elif lower_shadow > 2 * body:
            return 1
    return 0


def is_engulfing(df, i):
    if i < 1:
        return 0

    prev, curr = df.iloc[i-1], df.iloc[i]
    bull_engulf = (curr['close'] > curr['open'] and
                   prev['close'] < prev['open'] and
                   curr['open'] < prev['close'] and
                   curr['close'] > prev['open'] and
                   curr['volume'] > prev['volume'] * 0.8)

    bear_engulf = (curr['close'] < curr['open'] and
                   prev['close'] > prev['open'] and
                   curr['open'] > prev['close'] and
                   curr['close'] < prev['open'] and
                   curr['volume'] > prev['volume'] * 0.8)

    return 1 if bull_engulf else (-1 if bear_engulf else 0)


def detect_strong_selling_candles(df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[bool, float, int]:
    """
    Detect strong selling candles within a range

    Parameters:
    -----------
    df: DataFrame with price and volume data
    start_idx: Start index for analysis
    end_idx: End index for analysis

    Returns:
    --------
    Tuple[bool, float, int]: Whether strong selling is detected, strength score, and count of strong candles
    """
    if start_idx >= len(df) or end_idx >= len(df) or start_idx > end_idx:
        return False, 0, 0

    range_data = df.iloc[start_idx:end_idx+1]

    # Count bearish candles with significant bodies
    strong_bearish_count = 0
    total_volume_ratio = 0
    total_body_ratio = 0

    prev_candle = None
    for idx, candle in range_data.iterrows():
        is_bearish = candle['close'] < candle['open']

        if is_bearish:
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            body_ratio = body_size / candle_range if candle_range > 0 else 0

            # Check if this is a significant bearish candle
            if body_ratio > 0.5:  # Body is at least 50% of the candle range
                strong_bearish_count += 1
                total_body_ratio += body_ratio

                # Check volume if we have a previous candle
                if prev_candle is not None:
                    volume_ratio = candle['volume'] / \
                        prev_candle['volume'] if prev_candle['volume'] > 0 else 1
                    total_volume_ratio += volume_ratio

        prev_candle = candle

    # Calculate strength metrics
    avg_body_ratio = total_body_ratio / \
        strong_bearish_count if strong_bearish_count > 0 else 0
    avg_volume_ratio = total_volume_ratio / \
        (strong_bearish_count - 1) if strong_bearish_count > 1 else 0

    # Calculate overall strength score
    strength_score = 0

    # Base score on count of strong bearish candles
    if strong_bearish_count == 1:
        strength_score = 30
    elif strong_bearish_count == 2:
        strength_score = 60
    elif strong_bearish_count >= 3:
        strength_score = 80

    # Adjust score based on average body ratio
    if avg_body_ratio > 0.7:
        strength_score += 20
    elif avg_body_ratio > 0.5:
        strength_score += 10

    # Adjust score based on volume
    if avg_volume_ratio > 1.5:
        strength_score += 20
    elif avg_volume_ratio > 1.0:
        strength_score += 10

    # Cap score at 100
    strength_score = min(100, strength_score)

    # Determine if this is strong selling
    is_strong_selling = strength_score >= 50

    return is_strong_selling, strength_score, strong_bearish_count


def detect_strong_buying_candles(df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[bool, float, int]:
    """
    Detect strong buying candles within a range

    Parameters:
    -----------
    df: DataFrame with price and volume data
    start_idx: Start index for analysis
    end_idx: End index for analysis

    Returns:
    --------
    Tuple[bool, float, int]: Whether strong buying is detected, strength score, and count of strong candles
    """
    if start_idx >= len(df) or end_idx >= len(df) or start_idx > end_idx:
        return False, 0, 0

    range_data = df.iloc[start_idx:end_idx+1]

    # Count bullish candles with significant bodies
    strong_bullish_count = 0
    total_volume_ratio = 0
    total_body_ratio = 0

    prev_candle = None
    for idx, candle in range_data.iterrows():
        is_bullish = candle['close'] > candle['open']

        if is_bullish:
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            body_ratio = body_size / candle_range if candle_range > 0 else 0

            # Check if this is a significant bullish candle
            if body_ratio > 0.5:  # Body is at least 50% of the candle range
                strong_bullish_count += 1
                total_body_ratio += body_ratio

                # Check volume if we have a previous candle
                if prev_candle is not None:
                    volume_ratio = candle['volume'] / \
                        prev_candle['volume'] if prev_candle['volume'] > 0 else 1
                    total_volume_ratio += volume_ratio

        prev_candle = candle

    # Calculate strength metrics
    avg_body_ratio = total_body_ratio / \
        strong_bullish_count if strong_bullish_count > 0 else 0
    avg_volume_ratio = total_volume_ratio / \
        (strong_bullish_count - 1) if strong_bullish_count > 1 else 0

    # Calculate overall strength score
    strength_score = 0

    # Base score on count of strong bullish candles
    if strong_bullish_count == 1:
        strength_score = 30
    elif strong_bullish_count == 2:
        strength_score = 60
    elif strong_bullish_count >= 3:
        strength_score = 80

    # Adjust score based on average body ratio
    if avg_body_ratio > 0.7:
        strength_score += 20
    elif avg_body_ratio > 0.5:
        strength_score += 10

    # Adjust score based on volume
    if avg_volume_ratio > 1.5:
        strength_score += 20
    elif avg_volume_ratio > 1.0:
        strength_score += 10

    # Cap score at 100
    strength_score = min(100, strength_score)

    # Determine if this is strong buying
    is_strong_buying = strength_score >= 50

    return is_strong_buying, strength_score, strong_bullish_count


def detect_strong_reversal_signal(df: pd.DataFrame, current_index: int, direction: int) -> Tuple[bool, float]:
    """
    Detect strong reversal signals against a given direction

    Parameters:
    -----------
    df: DataFrame with price and volume data
    current_index: Current candle index
    direction: Expected direction (1 for bullish, -1 for bearish)

    Returns:
    --------
    Tuple[bool, float]: Whether a strong reversal is detected and confidence score
    """
    # Need at least 3 candles
    if current_index < 3:
        return False, 0

    # Look back further for pattern detection
    lookback = min(5, current_index)
    start_idx = current_index - lookback

    # For bullish OB, check for strong selling pattern (bearish reversal)
    if direction == 1:
        is_strong_selling, selling_strength, selling_count = detect_strong_selling_candles(
            df, start_idx, current_index)

        # If we detect strong selling pattern, return immediately with high confidence
        if is_strong_selling and selling_strength >= 70:
            return True, selling_strength

    # For bearish OB, check for strong buying pattern (bullish reversal)
    elif direction == -1:
        is_strong_buying, buying_strength, buying_count = detect_strong_buying_candles(
            df, start_idx, current_index)

        # If we detect strong buying pattern, return immediately with high confidence
        if is_strong_buying and buying_strength >= 70:
            return True, buying_strength

    # Evaluate last 3 candles for more nuanced signals
    recent_candles = df.iloc[current_index-3:current_index+1]

    # Counters for reversal signals
    counter_candles = 0  # Candles against direction
    strong_counter_body = 0  # Candles with strong bodies
    high_volume_signals = 0  # Candles with high volume

    # Analyze recent price action
    for i in range(len(recent_candles)):
        candle = recent_candles.iloc[i]

        # Basic candle properties
        is_bull = candle['close'] > candle['open']
        candle_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        rel_body_size = candle_size / candle_range if candle_range > 0 else 0

        # Check if candle direction is against our expected direction
        if (direction == 1 and not is_bull) or (direction == -1 and is_bull):
            counter_candles += 1

            # Strong body candle
            if rel_body_size > 0.6:
                strong_counter_body += 1

            # Check for high volume
            if i > 0:
                prev_volume = recent_candles.iloc[i-1]['volume']
                if candle['volume'] > prev_volume * 1.5:
                    high_volume_signals += 1

    # Calculate a confidence score for reversal
    confidence = 0

    # Basic reversal score based on counter candles
    if counter_candles >= 2:
        confidence += 30

    # Bonus for strong body candles
    confidence += strong_counter_body * 20

    # Bonus for high volume
    confidence += high_volume_signals * 15

    # Check additional technical indicators if available
    try:
        # Check MACD for momentum change
        macd = df['macd'].iloc[current_index]
        macd_signal = df['macd_signal'].iloc[current_index]
        macd_hist = df['macd_hist'].iloc[current_index]
        prev_hist = df['macd_hist'].iloc[current_index -
                                         1] if current_index > 0 else 0

        # MACD momentum against our direction
        if (direction == 1 and macd < macd_signal) or (direction == -1 and macd > macd_signal):
            confidence += 15

        # MACD histogram changing against our direction
        if (direction == 1 and macd_hist < prev_hist) or (direction == -1 and macd_hist > prev_hist):
            confidence += 15
    except:
        # MACD columns not available
        pass

    # Is this a strong enough reversal signal?
    is_reversal = confidence >= 50

    return is_reversal, confidence


def analyze_candle_volume(df: pd.DataFrame, current_index: int, lookback: int = 20) -> Dict:
    """
    Comprehensive function to analyze candle and volume patterns
    Combines the features of the original analyze_candle_volume and analyze_volume_patterns

    Parameters:
    -----------
    df: DataFrame with price and volume data
    current_index: Current candle index
    lookback: Number of periods to look back for average calculations

    Returns:
    --------
    Dict containing volume analysis, candle analysis, pressure, and buy/sell ratios
    """
    if current_index >= len(df):
        raise ValueError(
            f"current_index {current_index} is out of bounds for df with length {len(df)}")

    # Extract recent data
    recent_data = df.iloc[max(0, current_index-lookback+1)
                              :current_index+1].copy()
    current_candle = df.iloc[current_index]
    prev_candle = df.iloc[current_index-1] if current_index > 0 else None

    # Calculate Volume RSI
    volume_rsi = calculate_rsi(recent_data['volume'], rsi_length=14)
    current_volume_rsi = volume_rsi.iloc[-1] if not volume_rsi.empty else None

    # Basic candle properties
    is_bullish = current_candle['close'] > current_candle['open']
    volume_increased = False
    if prev_candle is not None:
        volume_increased = current_candle['volume'] > prev_candle['volume']

    # Calculate candle body and wicks
    body_size = abs(current_candle['close'] - current_candle['open'])
    upper_wick = current_candle['high'] - \
        max(current_candle['open'], current_candle['close'])
    lower_wick = min(
        current_candle['open'], current_candle['close']) - current_candle['low']
    candle_range = current_candle['high'] - current_candle['low']
    relative_body_size = body_size / candle_range if candle_range > 0 else 0

    # Analyze recent candle patterns for buy/sell ratios
    recent_candles = recent_data.tail(min(5, len(recent_data)))
    bullish_candles = recent_candles[recent_candles['close']
                                     > recent_candles['open']]
    bearish_candles = recent_candles[recent_candles['close']
                                     <= recent_candles['open']]

    buy_volume = bullish_candles['volume'].sum()
    sell_volume = bearish_candles['volume'].sum()
    total_volume = buy_volume + sell_volume

    # Calculate buy/sell ratios
    buy_ratio = (buy_volume / total_volume * 100) if total_volume > 0 else 0
    sell_ratio = (sell_volume / total_volume * 100) if total_volume > 0 else 0

    # Calculate volume metrics
    avg_volume = recent_data['volume'].mean()
    current_volume = current_candle['volume']
    volume_ratio = current_volume / avg_volume

    # Volume trend calculation
    volume_trend = recent_data['volume'].pct_change().mean() * 100

    # Volume score calculation (comprehensive)
    base_volume_score = 0

    # Base on volume ratio
    if volume_ratio > 2.0:
        base_volume_score = 100 if volume_increased else 90
    elif volume_ratio > 1.5:
        base_volume_score = 80 if volume_increased else 70
    elif volume_ratio > 1.2:
        base_volume_score = 60 if volume_increased else 50
    elif volume_ratio > 1.0:
        base_volume_score = 35 if volume_increased else 25
    else:
        base_volume_score = 10 if volume_increased else 0

    # Adjust with trend and ratios
    trend_adjusted_volume_score = base_volume_score
    if volume_trend > 0:
        trend_adjusted_volume_score += 10  # Positive trend bonus
    if buy_ratio > 60 and is_bullish:
        trend_adjusted_volume_score = min(
            100, trend_adjusted_volume_score + 15)  # Strong buying bonus
    elif sell_ratio > 60 and not is_bullish:
        trend_adjusted_volume_score = min(
            100, trend_adjusted_volume_score + 15)  # Strong selling bonus

    # Analyze previous candle trend
    prev_trend = None
    if prev_candle is not None:
        prev_trend = "bullish" if prev_candle['close'] > prev_candle['open'] else "bearish"

    # Analyze trend change
    trend_change = None
    if prev_candle is not None:
        if is_bullish and prev_trend == "bearish":
            trend_change = "bullish_reversal"
        elif not is_bullish and prev_trend == "bullish":
            trend_change = "bearish_reversal"

    # Classify candle type (from analyze_volume_patterns)
    if is_bullish:
        if body_size > upper_wick and volume_increased:
            candle_type = "STRONG_BUY"
            candle_score = 90
        elif body_size > upper_wick:
            candle_type = "BUY"
            candle_score = 70
        elif upper_wick > body_size:
            candle_type = "WEAK_BUY"
            candle_score = 55
        else:
            candle_type = "NEUTRAL"
            candle_score = 50
    else:
        if body_size > lower_wick and volume_increased:
            candle_type = "STRONG_SELL"
            candle_score = 10
        elif body_size > lower_wick:
            candle_type = "SELL"
            candle_score = 30
        elif lower_wick > body_size:
            candle_type = "WEAK_SELL"
            candle_score = 45
        else:
            candle_type = "NEUTRAL"
            candle_score = 50

    # Determine pressure with unified approach
    if is_bullish:
        if volume_ratio > 1.5 and volume_increased and buy_ratio > 60:
            pressure = "Strong Buying Pressure"
            pressure_score = 90
        elif volume_ratio > 1.2 or (volume_increased and candle_type in ["STRONG_BUY", "BUY"]):
            pressure = "Moderate Buying Pressure"
            pressure_score = 70
        else:
            pressure = "Weak Buying Pressure"
            pressure_score = 60
    else:  # Bearish
        if volume_ratio > 1.5 and volume_increased and sell_ratio > 60:
            pressure = "Strong Selling Pressure"
            pressure_score = 10
        elif volume_ratio > 1.2 or (volume_increased and candle_type in ["STRONG_SELL", "SELL"]):
            pressure = "Moderate Selling Pressure"
            pressure_score = 30
        else:
            pressure = "Weak Selling Pressure"
            pressure_score = 40

    # Add climax detection
    if is_bullish and volume_ratio > 2.0 and relative_body_size > 0.7 and trend_change == "bullish_reversal":
        pressure = "Strong Buying Climax"
        pressure_score = 95
    elif not is_bullish and volume_ratio > 2.0 and relative_body_size > 0.7 and trend_change == "bearish_reversal":
        pressure = "Strong Selling Climax"
        pressure_score = 5

    # Create comprehensive return structure compatible with both functions
    return {
        'volume_analysis': {
            'volume_ratio': volume_ratio,
            'volume_score': trend_adjusted_volume_score,
            'volume_rsi': current_volume_rsi,
            'volume_increased': volume_increased,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'volume_trend': volume_trend
        },
        'candle_analysis': {
            'is_bullish': is_bullish,
            'body_size': body_size,
            'relative_body_size': relative_body_size,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'trend_change': trend_change,
            'type': candle_type,
            'score': candle_score
        },
        'pressure': {
            'type': pressure,
            'score': pressure_score
        },
        'recent_pattern': {
            'bullish_count': len(bullish_candles),
            'bearish_count': len(bearish_candles),
            'dominant_side': 'BULLISH' if len(bullish_candles) > len(bearish_candles) else 'BEARISH'
        },
        # For backward compatibility with analyze_volume_patterns
        'volume_rsi': current_volume_rsi,
        'buy_ratio': buy_ratio,
        'volume_score': trend_adjusted_volume_score,
        'pressure_ratio': pressure_score / 100,
        'sell_ratio': sell_ratio,
        'volume_trend': volume_trend,
        'analysis': {
            'pressure': pressure,
            'score': pressure_score,
        },
        'last_candle': {
            'type': candle_type,
            'score': candle_score,
            'volume': current_volume,
            'volume_ratio': volume_ratio,
            'body_size': body_size,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick
        }
    }


def should_keep_ob(df: pd.DataFrame, ob: Dict, current_index: int, use_should_keep_ob: bool = True, analysis: Dict = None) -> Tuple[bool, Dict]:
    """
    Enhanced unified function to evaluate order blocks with early detection of strong reversals

    Parameters:
    -----------
    df: DataFrame with price and volume data
    ob: Dictionary containing order block information
    current_index: Current candle index
    use_should_keep_ob: Flag to determine whether to evaluate OB (set to False to always return True)
    analysis: Pre-computed candle and volume analysis

    Returns:
    --------
    Tuple[bool, Dict]: Decision to keep OB and dictionary with detailed analysis results
    """
    if not use_should_keep_ob:
        return True, {"setup_quality": 100, "final_score": 100, "threshold": 0, "warnings": []}

    ob_direction = ob['direction']
    ob_direction_str = "BULLISH" if ob_direction == 1 else "BEARISH"
    analysis = analysis or analyze_candle_volume(df, current_index)

    # Get pin bar and engulfing signals
    pin_bar = is_pin_bar(df.iloc[current_index])
    engulfing = is_engulfing(df, current_index)

    # Create warnings list for tracking issues
    warnings = []

    # Check for strong reversal signal against OB direction
    is_reversal, reversal_confidence = detect_strong_reversal_signal(
        df, current_index, ob_direction)

    # Check for specific reversal pattern to provide more detailed information
    reversal_type = "unknown"
    if ob_direction == 1:  # Bullish OB
        is_strong_selling, selling_strength, selling_count = detect_strong_selling_candles(
            df, max(0, current_index-3), current_index)
        if is_strong_selling and selling_strength >= 60:
            reversal_type = f"strong selling ({selling_count} bearish candles, strength: {selling_strength:.1f}%)"
    else:  # Bearish OB
        is_strong_buying, buying_strength, buying_count = detect_strong_buying_candles(
            df, max(0, current_index-3), current_index)
        if is_strong_buying and buying_strength >= 60:
            reversal_type = f"strong buying ({buying_count} bullish candles, strength: {buying_strength:.1f}%)"

    # Immediately reject OB if strong reversal is detected with high confidence
    if is_reversal and reversal_confidence >= 70:
        print(
            f"🔴 REJECTING {ob_direction_str} OB: Strong reversal detected with {reversal_confidence:.1f}% confidence")
        print(f"   Reversal type: {reversal_type}")
        warnings.append(
            f"Strong reversal detected ({reversal_confidence:.1f}% confidence)")
        warnings.append(f"Reversal pattern: {reversal_type}")
        return False, {
            "setup_quality": 0,
            "final_score": 0,
            "threshold": 70,
            "warnings": warnings,
            "reversal_detected": True,
            "reversal_confidence": reversal_confidence,
            "reversal_type": reversal_type
        }

    # Check for strong momentum against OB direction from recent candles
    momentum_against_ob = False
    strong_volume_against_ob = False

    # Check last 3 candles for strong counter-momentum
    lookback = min(3, current_index)
    strong_counter_candles = 0
    strong_volume_candles = 0

    if lookback > 0:
        # Get recent candles
        recent_candles = df.iloc[current_index-lookback:current_index+1]

        # Count strong bearish/bullish candles against OB direction
        for i in range(len(recent_candles)):
            candle = recent_candles.iloc[i]
            is_bull = candle['close'] > candle['open']
            candle_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            rel_body_size = candle_size / candle_range if candle_range > 0 else 0

            # Check for candles that oppose OB direction
            if (ob_direction == 1 and not is_bull) or (ob_direction == -1 and is_bull):
                # If candle has large body relative to range
                if rel_body_size > 0.6:
                    strong_counter_candles += 1

                # Check for high volume
                if i > 0:
                    prev_volume = recent_candles.iloc[i-1]['volume']
                    if candle['volume'] > prev_volume * 1.5:
                        strong_volume_candles += 1

        # Detect momentum against OB
        momentum_against_ob = strong_counter_candles >= 2
        strong_volume_against_ob = strong_volume_candles >= 2

    # Immediately reject OB if strong counter-momentum is detected
    if momentum_against_ob and strong_volume_against_ob:
        counter_direction = "SELLING" if ob_direction == 1 else "BUYING"
        print(
            f"🔴 REJECTING {ob_direction_str} OB: Strong {counter_direction} momentum detected with {strong_counter_candles} candles and high volume")
        warning_msg = f"Strong {counter_direction} momentum with {strong_counter_candles} candles and high volume"
        warnings.append(warning_msg)
        return False, {
            "setup_quality": 0,
            "final_score": 0,
            "threshold": 70,
            "warnings": warnings,
            "strong_counter_momentum": True,
            "counter_candles": strong_counter_candles,
            "counter_direction": counter_direction
        }

    # Price Action Score (40%)
    price_action_score = 0

    # Basic price action signals
    if ob_direction == 1:  # Bullish
        if pin_bar == 1:
            price_action_score += 40
        if engulfing == 1:
            price_action_score += 40
        if df.iloc[current_index]['low'] <= ob['bottom']:
            price_action_score += 20

        # Add bonus for bullish trend change
        if analysis['candle_analysis'].get('trend_change') == 'bullish_reversal':
            price_action_score = min(100, price_action_score + 20)
    else:  # Bearish
        if pin_bar == -1:
            price_action_score += 40
        if engulfing == -1:
            price_action_score += 40
        if df.iloc[current_index]['high'] >= ob['top']:
            price_action_score += 20

        # Add bonus for bearish trend change
        if analysis['candle_analysis'].get('trend_change') == 'bearish_reversal':
            price_action_score = min(100, price_action_score + 20)

    # Apply penalty for reversal signals
    if is_reversal:
        # Apply penalty proportional to reversal confidence
        penalty_factor = reversal_confidence / 100
        price_action_score = max(0, price_action_score - (40 * penalty_factor))
        warnings.append(
            f"Reversal signal present ({reversal_confidence:.1f}% confidence)")

    # Momentum Score (30%)
    momentum_score = 0
    try:
        macd = df['macd'].iloc[current_index]
        macd_signal = df['macd_signal'].iloc[current_index]
        macd_hist = df['macd_hist'].iloc[current_index]
        prev_hist = df['macd_hist'].iloc[current_index -
                                         1] if current_index > 0 else 0

        if ob_direction == 1:  # Bullish
            if macd > macd_signal:
                momentum_score += 50
            if macd_hist > prev_hist:
                momentum_score += 50
        else:  # Bearish
            if macd < macd_signal:
                momentum_score += 50
            if macd_hist < prev_hist:
                momentum_score += 50
    except:
        # MACD columns may not be available
        momentum_score = 50  # Neutral score if MACD not available

    # Apply penalty for reversal momentum
    if is_reversal:
        # Apply penalty proportional to reversal confidence
        penalty_factor = reversal_confidence / 100
        momentum_score = max(0, momentum_score - (50 * penalty_factor))

    # Volume Score (30%) - using the enhanced analysis
    volume_score = analysis['volume_analysis']['volume_score']

    # Pressure alignment bonus
    pressure_type = analysis['pressure']['type']
    if (ob_direction == 1 and "Buying" in pressure_type) or (ob_direction == -1 and "Selling" in pressure_type):
        # Bonus for aligned pressure
        if "Strong" in pressure_type:
            volume_score = min(100, volume_score + 20)
        elif "Moderate" in pressure_type:
            volume_score = min(100, volume_score + 10)

    # Penalty for opposing pressure
    elif (ob_direction == 1 and "Selling" in pressure_type) or (ob_direction == -1 and "Buying" in pressure_type):
        if "Strong" in pressure_type:
            volume_score = max(0, volume_score - 40)
            momentum_score = max(0, momentum_score - 30)
            warnings.append(f"{pressure_type} against order block direction")
        elif "Moderate" in pressure_type:
            volume_score = max(0, volume_score - 20)
            momentum_score = max(0, momentum_score - 15)
            warnings.append(f"{pressure_type} against order block direction")

    # Calculate final score with potentially more weights for critical factors
    final_score = (
        price_action_score * 0.4 +
        momentum_score * 0.3 +
        volume_score * 0.3
    )

    # Adaptive threshold based on market conditions
    threshold = 35

    # Increase threshold if there are any counter signals
    if strong_counter_candles >= 1 or is_reversal or ("Strong" in pressure_type and not ((ob_direction == 1 and "Buying" in pressure_type) or (ob_direction == -1 and "Selling" in pressure_type))):
        # Calculate dynamic threshold based on reversal strength
        if is_reversal:
            # Scale from 60 to 80 based on reversal confidence
            threshold = 60 + (reversal_confidence / 100) * 20
        else:
            threshold = 60  # Base higher threshold when there's any sign of counter momentum

        if strong_counter_candles >= 1:
            warnings.append(
                f"{strong_counter_candles} counter trend candles detected")

    # Map final_score to setup_quality scale (0-100)
    # We want to keep the same scale for compatibility
    setup_quality = final_score

    # Determine setup strength based on setup_quality
    if setup_quality >= 80:
        setup_strength = "Strong"
    elif setup_quality >= 65:
        setup_strength = "Moderate"
    elif setup_quality >= 50:
        setup_strength = "Weak"
    else:
        setup_strength = "Very Weak"

    # Determine entry quality
    if setup_quality >= 80:
        entry_quality = "Excellent"
    elif setup_quality >= 65:
        entry_quality = "Good"
    elif setup_quality >= 50:
        entry_quality = "Moderate"
    else:
        entry_quality = "Poor"

    # Determine risk level
    if setup_quality >= 75:
        risk_level = "Low"
    elif setup_quality >= 60:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Debug info for accepted OBs
    if final_score >= threshold:
        print(f"✅ KEEPING {ob_direction_str} OB: Score: {final_score:.1f} - Price: {price_action_score:.1f}, Momentum: {momentum_score:.1f}, Volume: {volume_score:.1f}, Threshold: {threshold:.1f}")
    else:
        print(f"🔴 REJECTING {ob_direction_str} OB: Low score {final_score:.1f} < {threshold:.1f} - Price: {price_action_score:.1f}, Momentum: {momentum_score:.1f}, Volume: {volume_score:.1f}")
        if is_reversal:
            print(
                f"   Reason: Reversal signal with {reversal_confidence:.1f}% confidence")
            if reversal_type != "unknown":
                print(f"   Pattern: {reversal_type}")
        if strong_counter_candles > 0:
            print(
                f"   Reason: {strong_counter_candles} strong counter candles detected")
        if "Strong" in pressure_type and ((ob_direction == 1 and "Selling" in pressure_type) or (ob_direction == -1 and "Buying" in pressure_type)):
            print(f"   Reason: {pressure_type} against OB direction")

    # Create comprehensive result dictionary
    result = {
        "setup_quality": setup_quality,
        "final_score": final_score,
        "threshold": threshold,
        "keep_ob": final_score >= threshold,
        "scores": {
            "price_action": price_action_score,
            "momentum": momentum_score,
            "volume": volume_score
        },
        "strength": setup_strength,
        "entry_quality": entry_quality,
        "risk_level": risk_level,
        "warnings": warnings,
        "pressure": {
            "type": pressure_type,
            "score": analysis['pressure']['score']
        },
        "reversal": {
            "detected": is_reversal,
            "confidence": reversal_confidence if is_reversal else 0,
            "type": reversal_type if is_reversal and reversal_type != "unknown" else None
        },
        "counter_momentum": {
            "detected": momentum_against_ob,
            "strong_volume": strong_volume_against_ob,
            "counter_candles": strong_counter_candles
        }
    }

    # Return both decision and detailed results
    return final_score >= threshold, result
