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
    recent_data = df.iloc[max(0, current_index-lookback+1)                          :current_index+1].copy()
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


def should_keep_ob(df: pd.DataFrame, ob: Dict, current_index: int, use_should_keep_ob: bool = True) -> Tuple[bool, float]:
    """
    Enhanced unified function to evaluate order blocks

    Parameters:
    -----------
    df: DataFrame with price and volume data
    ob: Dictionary containing order block information
    current_index: Current candle index
    use_should_keep_ob: Flag to determine whether to evaluate OB (set to False to always return True)

    Returns:
    --------
    Tuple[bool, float]: Decision to keep OB and confidence score
    """
    if not use_should_keep_ob:
        return True, 100

    ob_direction = ob['direction']

    # Get enhanced volume and candle analysis
    analysis = analyze_candle_volume(df, current_index)

    # Get pin bar and engulfing signals
    pin_bar = is_pin_bar(df.iloc[current_index])
    engulfing = is_engulfing(df, current_index)

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

    # Momentum Score (30%)
    momentum_score = 0
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

    # Calculate final score with potentially more weights for critical factors
    final_score = (
        price_action_score * 0.4 +
        momentum_score * 0.3 +
        volume_score * 0.3
    )

    # Debug info
    if final_score >= 35:
        print(
            f"OB Score: {final_score:.2f} - Price: {price_action_score}, Momentum: {momentum_score}, Volume: {volume_score}")

    # Return both decision and score
    return final_score >= 35, final_score
