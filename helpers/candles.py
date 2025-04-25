import pandas as pd
from typing import Dict, Tuple

# bullish pattern


def is_hammer(candle):
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    return (body < 0.3 * range_) and (lower_shadow > 2 * body) and (upper_shadow < 0.3 * body)


def is_morning_star(candle1, candle2, candle3):
    if candle1 is None or candle2 is None:
        return False
    body1 = abs(candle1['close'] - candle1['open'])
    body2 = abs(candle2['close'] - candle2['open'])
    body3 = abs(candle3['close'] - candle3['open'])
    return (candle1['close'] < candle1['open']) and \
           (body2 < 0.3 * (candle2['high'] - candle2['low'])) and \
           (candle3['close'] > candle3['open']) and \
           (candle3['close'] > candle1['open'] - body1 / 2)


# bearish pattern
def is_shooting_star(candle):
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    return (body < 0.3 * range_) and (upper_shadow > 2 * body) and (lower_shadow < 0.3 * body)


def is_evening_star(candle1, candle2, candle3):
    if candle1 is None or candle2 is None:
        return False
    body1 = abs(candle1['close'] - candle1['open'])
    body2 = abs(candle2['close'] - candle2['open'])
    body3 = abs(candle3['close'] - candle3['open'])
    return (candle1['close'] > candle1['open']) and \
           (body2 < 0.3 * (candle2['high'] - candle2['low'])) and \
           (candle3['close'] < candle3['open']) and \
           (candle3['close'] < candle1['open'] + body1 / 2)

   # Helper function to detect Pin Bar candles


def is_pin_bar(candle):
    """Identify Pin Bar candles (1 = Bullish, -1 = Bearish, 0 = None)"""
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    if range_ > 0 and body < 0.1 * range_:
        if upper_shadow > 2 * body:
            return -1  # Bearish pin bar
        elif lower_shadow > 2 * body:
            return 1   # Bullish pin bar
    return 0

    # Helper function to detect Engulfing patterns


def is_engulfing(prev, curr):
    """Identify Engulfing patterns (1 = Bullish, -1 = Bearish, 0 = None)"""
    if prev is None:
        return 0
    if (curr['close'] > curr['open'] and prev['close'] < prev['open'] and
            curr['open'] < prev['close'] and curr['close'] > prev['open']):
        return 1  # Bullish engulfing
    elif (curr['close'] < curr['open'] and prev['close'] > prev['open'] and
          curr['open'] > prev['close'] and curr['close'] < prev['open']):
        return -1  # Bearish engulfing
    return 0


def is_doji(candle, body_ratio=0.1):
    """Check if a candle is a Doji (small body relative to range)."""
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    return body < body_ratio * range_


def should_keep_ob(df: pd.DataFrame, ob: Dict, current_index: int, use_should_keep_ob: bool = True, analysis: Dict = None, strength_threshold: int = 70) -> Tuple[bool, Dict]:
    """
    Optimized function to evaluate order blocks, balancing strictness and flexibility to reduce missed opportunities.

    Parameters:
    -----------
    df: DataFrame with price (open, high, low, close), volume, and MACD data
    ob: Dict with OB details (e.g., 'direction': 1/-1, 'top', 'bottom')
    current_index: Index of the current candle
    use_should_keep_ob: If False, always keep OB (bypasses evaluation)
    analysis: Optional pre-computed analysis (volume, pressure)

    Returns:
    --------
    Tuple[bool, Dict]: (keep_ob, result_dict) where result_dict contains:
        - setup_quality: Overall quality score (0-100)
        - final_score: Weighted score (0-100)
        - threshold: Adaptive decision threshold
        - warnings: List of issues detected
        - Additional metrics (scores, strength, etc.)
    """
    # If evaluation is bypassed, return True with perfect scores
    if not use_should_keep_ob:
        return True, {"setup_quality": 100, "final_score": 100, "threshold": 0, "warnings": []}

    # Extract OB direction and set string representation
    ob_direction = ob['direction']  # 1 = Bullish, -1 = Bearish
    ob_direction_str = "BULLISH" if ob_direction == 1 else "BEARISH"
    analysis = analysis or {'volume_analysis': {'volume_score': 50}, 'pressure': {
        'type': 'Neutral'}}  # Default if not provided

    # Get current and previous candles
    current_candle = df.iloc[current_index]
    prev_candle = df.iloc[current_index - 1] if current_index > 0 else None
    pin_bar = is_pin_bar(current_candle)
    engulfing = is_engulfing(prev_candle, current_candle)
    if ob['strength'] < strength_threshold:
        return False, {
            "setup_quality": 0,
            "final_score": 0,
            "threshold": 75,
            "warnings": ["OB strength is too weak"],
            "reversal_score": 0
        }
    reversal_score = 0
    warnings = []
    lookback = min(16, current_index)
    if lookback > 0:
        recent_candles = df.iloc[current_index - lookback:current_index + 1]
        for i in range(len(recent_candles)):
            candle = recent_candles.iloc[i]
            prev_candle = recent_candles.iloc[i - 1] if i > 0 else None
            prev_prev_candle = recent_candles.iloc[i - 2] if i > 1 else None

        # Detect reversal patterns
            doji = is_doji(candle)
            hammer = is_hammer(candle)
            morning_star = is_morning_star(
                prev_prev_candle, prev_candle, candle) if (prev_prev_candle is not None and prev_candle is not None) else False
            pin_bar = is_pin_bar(candle)
            engulfing = is_engulfing(prev_candle, candle)

            if ob_direction == 1:  # Bullish OB, look for bearish reversal patterns
                if pin_bar == -1:
                    # Bearish Pin Bar
                    reversal_score += 25
                if engulfing == -1:
                    # Bearish Engulfing (strong signal)
                    reversal_score += 25
                if doji:
                    # Doji (weaker signal)
                    reversal_score += 10

            elif ob_direction == -1:  # Bearish OB, look for bullish reversal patterns
                if pin_bar == 1:
                    # Bullish Pin Bar
                    reversal_score += 25
                if engulfing == 1:
                    # Bullish Engulfing (strong signal)
                    reversal_score += 25
                if hammer:
                    # Hammer (bullish reversal)
                    reversal_score += 25
                if morning_star:
                    # Morning Star (strong bullish reversal)
                    reversal_score += 30
                if doji:
                    reversal_score += 10  # Doji (weaker signal)

    # Add MACD-based reversal signals
    try:
        macd, macd_signal, macd_hist = df['macd'].iloc[current_index], df[
            'macd_signal'].iloc[current_index], df['macd_hist'].iloc[current_index]
        prev_hist = df['macd_hist'].iloc[current_index -
                                         1] if current_index > 0 else 0
        if (ob_direction == 1 and macd < macd_signal) or (ob_direction == -1 and macd > macd_signal):
            # MACD crossover against OB direction
            reversal_score += 25
        if (ob_direction == 1 and macd_hist < prev_hist) or (ob_direction == -1 and macd_hist > prev_hist):
            # MACD histogram weakening
            reversal_score += 25
    except KeyError:
        pass
    try:
        ema34 = df['ema34'].iloc[current_index]
        ema89 = df['ema89'].iloc[current_index]
        prev_ema34 = df['ema34'].iloc[current_index -
                                      1] if current_index > 0 else None
        prev_ema89 = df['ema89'].iloc[current_index -
                                      1] if current_index > 0 else None
        close = df['close'].iloc[current_index]
        prev_close = df['close'].iloc[current_index -
                                      1] if current_index > 0 else None
        if prev_ema34 is not None and prev_ema89 is not None and prev_close is not None:
            if ob_direction == 1:
                # Check for EMA crossover: EMA34 crosses below EMA89
                if prev_ema34 > prev_ema89 and ema34 < ema89:
                    reversal_score += 30
                # Check for price crossing below EMA34
                if prev_close > prev_ema34 and close < ema34:
                    reversal_score += 20
                # Check for EMA convergence
                strength_previous = prev_ema34 - prev_ema89
                strength_current = ema34 - ema89
                if strength_current < strength_previous and strength_previous > 0:
                    reversal_score += 15
            elif ob_direction == -1:
                # Check for EMA crossover: EMA34 crosses above EMA89
                if prev_ema34 < prev_ema89 and ema34 > ema89:
                    reversal_score += 30
                # Check for price crossing above EMA34
                if prev_close < prev_ema34 and close > ema34:
                    reversal_score += 20
                # Check for EMA convergence
                strength_previous = prev_ema89 - prev_ema34
                strength_current = ema89 - ema34
                if strength_current < strength_previous and strength_previous > 0:
                    reversal_score += 15
    except KeyError:
        pass
    reversal_score = min(reversal_score, 100)  # Cap reversal score at 100

    # Early rejection if reversal score is very high
    if reversal_score >= 75:
        print(
            f"🔴 REJECTING {ob_direction_str} OB: Strong reversal (score: {reversal_score})")
        warnings.append(f"Strong reversal (score: {reversal_score})")
        return False, {
            "setup_quality": 0,
            "final_score": 0,
            "threshold": 75,
            "warnings": warnings,
            "reversal_score": reversal_score
        }

    # Calculate Price Action Score
    price_action_score = 0
    if ob_direction == 1:  # Bullish OB
        if pin_bar == 1:
            price_action_score += 40  # Bullish Pin Bar
        if engulfing == 1:
            price_action_score += 40  # Bullish Engulfing
        if current_candle['low'] <= ob['bottom']:
            price_action_score += 20  # Price respects OB bottom
    else:  # Bearish OB
        if pin_bar == -1:
            price_action_score += 40  # Bearish Pin Bar
        if engulfing == -1:
            price_action_score += 40  # Bearish Engulfing
        if current_candle['high'] >= ob['top']:
            price_action_score += 20  # Price respects OB top

    # Calculate Momentum Score
    momentum_score = 50  # Neutral default
    try:
        if ob_direction == 1:
            if macd > macd_signal:
                momentum_score += 25  # Bullish MACD crossover
            if macd_hist > prev_hist:
                momentum_score += 25  # Increasing momentum
        else:
            if macd < macd_signal:
                momentum_score += 25  # Bearish MACD crossover
            if macd_hist < prev_hist:
                momentum_score += 25  # Increasing momentum
    except KeyError:
        pass  # Skip if MACD data is unavailable

    # Compute Final Score with Weights
    weights = {'price_action': 0.4, 'momentum': 0.3, 'volume': 0.3}
    # print()
    final_score = (
        price_action_score * weights['price_action'] +
        momentum_score * weights['momentum'] +
        ob['strength'] * weights['volume']
    )
    # Set Adaptive Threshold
    base_threshold = 40
    threshold = min(75, base_threshold + reversal_score * 0.6)

    # Determine if OB should be kept
    setup_quality = final_score
    keep_ob = final_score >= threshold

    # Additional Metrics
    setup_strength = "Strong" if setup_quality >= 80 else "Moderate" if setup_quality >= 65 else "Weak" if setup_quality >= 50 else "Very Weak"
    entry_quality = "Excellent" if setup_quality >= 80 else "Good" if setup_quality >= 65 else "Moderate" if setup_quality >= 50 else "Poor"
    risk_level = "Low" if setup_quality >= 75 else "Moderate" if setup_quality >= 60 else "High"

    # Compile Result Dictionary
    result = {
        "setup_quality": setup_quality,
        "final_score": final_score,
        "threshold": threshold,
        "keep_ob": keep_ob,
        "scores": {
            "price_action": price_action_score,
            "momentum": momentum_score,
            "volume": ob['strength']
        },
        "strength": setup_strength,
        "entry_quality": entry_quality,
        "risk_level": risk_level,
        "warnings": warnings,
        "reversal_score": reversal_score
    }

    # Debug Output
    if keep_ob:
        print(
            f"✅ KEEPING {ob_direction_str} OB: Score: {final_score:.1f} - Price: {price_action_score:.1f}, Momentum: {momentum_score:.1f}, Volume: {ob['strength']:.1f}, Threshold: {threshold:.1f}")
    else:
        print(
            f"🔴 REJECTING {ob_direction_str} OB: Score {final_score:.1f} < {threshold:.1f}")
        if reversal_score > 0:
            print(f"   Reversal score: {reversal_score}")

    return keep_ob, result
