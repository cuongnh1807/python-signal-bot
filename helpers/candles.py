import pandas as pd
from typing import Dict, Tuple


def should_keep_ob(df: pd.DataFrame, ob: Dict, current_index: int, use_should_keep_ob: bool = True, analysis: Dict = None) -> Tuple[bool, Dict]:
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

    # Get current and previous candles
    current_candle = df.iloc[current_index]
    prev_candle = df.iloc[current_index - 1] if current_index > 0 else None
    pin_bar = is_pin_bar(current_candle)
    engulfing = is_engulfing(prev_candle, current_candle)

    # Calculate reversal score based on recent candles
    reversal_score = 0
    warnings = []
    lookback = min(3, current_index)
    if lookback > 0:
        recent_candles = df.iloc[current_index - lookback:current_index + 1]
        counter_candles = 0
        high_volume_counter = 0
        for i in range(len(recent_candles)):
            candle = recent_candles.iloc[i]
            is_bull = candle['close'] > candle['open']
            if (ob_direction == 1 and not is_bull) or (ob_direction == -1 and is_bull):
                counter_candles += 1
                if i > 0 and candle['volume'] > recent_candles.iloc[i - 1]['volume'] * 1.5:
                    high_volume_counter += 1
        reversal_score += counter_candles * 15  # 15 points per counter candle
        # 15 points per high volume counter candle
        reversal_score += high_volume_counter * 15
        if high_volume_counter >= 2:
            warnings.append(
                f"High volume counter candles: {high_volume_counter}")

    # Add MACD-based reversal signals
    try:
        macd, macd_signal, macd_hist = df['macd'].iloc[current_index], df[
            'macd_signal'].iloc[current_index], df['macd_hist'].iloc[current_index]
        prev_hist = df['macd_hist'].iloc[current_index -
                                         1] if current_index > 0 else 0
        if (ob_direction == 1 and macd < macd_signal) or (ob_direction == -1 and macd > macd_signal):
            reversal_score += 20  # MACD crossover against OB direction
        if (ob_direction == 1 and macd_hist < prev_hist) or (ob_direction == -1 and macd_hist > prev_hist):
            reversal_score += 20  # MACD histogram weakening
    except KeyError:
        pass  # Skip if MACD data is unavailable

    # Integrate Pin Bar and Engulfing into reversal score
    if (ob_direction == 1 and pin_bar == -1) or (ob_direction == -1 and pin_bar == 1):
        reversal_score += 20  # Opposite Pin Bar detected
    if (ob_direction == 1 and engulfing == -1) or (ob_direction == -1 and engulfing == 1):
        reversal_score += 25  # Opposite Engulfing detected

    reversal_score = min(reversal_score, 100)  # Cap reversal score at 100

    # Early rejection if reversal score is very high
    if reversal_score >= 80:
        print(
            f"🔴 REJECTING {ob_direction_str} OB: Strong reversal (score: {reversal_score})")
        warnings.append(f"Strong reversal (score: {reversal_score})")
        return False, {
            "setup_quality": 0,
            "final_score": 0,
            "threshold": 80,
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

    # Calculate Volume Score with Pressure
    volume_score = analysis['volume_analysis']['volume_score']
    pressure_type = analysis['pressure']['type']
    if (ob_direction == 1 and "Buying" in pressure_type) or (ob_direction == -1 and "Selling" in pressure_type):
        if "Strong" in pressure_type:
            # Boost for strong pressure
            volume_score = min(100, volume_score + 20)
        elif "Moderate" in pressure_type:
            # Boost for moderate pressure
            volume_score = min(100, volume_score + 10)
    else:
        if "Strong" in pressure_type:
            # Penalty for strong opposite pressure
            volume_score = max(0, volume_score - 40)
        elif "Moderate" in pressure_type:
            # Penalty for moderate opposite pressure
            volume_score = max(0, volume_score - 20)

    # Compute Final Score with Weights
    weights = {'price_action': 0.4, 'momentum': 0.3, 'volume': 0.3}
    final_score = (
        price_action_score * weights['price_action'] +
        momentum_score * weights['momentum'] +
        volume_score * weights['volume']
    )

    # Set Adaptive Threshold
    base_threshold = 40
    threshold = min(80, base_threshold + (reversal_score / 100) * 40)

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
            "volume": volume_score
        },
        "strength": setup_strength,
        "entry_quality": entry_quality,
        "risk_level": risk_level,
        "warnings": warnings,
        "reversal_score": reversal_score
    }

    # Debug Output
    if keep_ob:
        print(f"✅ KEEPING {ob_direction_str} OB: Score: {final_score:.1f} - Price: {price_action_score:.1f}, Momentum: {momentum_score:.1f}, Volume: {volume_score:.1f}, Threshold: {threshold:.1f}")
    else:
        print(
            f"🔴 REJECTING {ob_direction_str} OB: Score {final_score:.1f} < {threshold:.1f}")
        if reversal_score > 0:
            print(f"   Reversal score: {reversal_score}")

    return keep_ob, result
