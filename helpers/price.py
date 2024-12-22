import pandas as pd


def analyze_candle_momentum(candles: pd.DataFrame) -> dict:
    """Analyze momentum based on recent candlestick patterns"""
    last_candle = candles.iloc[-1]
    prev_candle = candles.iloc[-2]

    # Calculate candle properties
    body_size = abs(last_candle['close'] - last_candle['open'])
    upper_wick = last_candle['high'] - \
        max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'],
                     last_candle['close']) - last_candle['low']

    # Determine candle strength
    is_bullish = last_candle['close'] > last_candle['open']
    body_to_wick_ratio = body_size / \
        (upper_wick + lower_wick + 0.0001)  # Avoid div by zero

    score = 50  # Neutral base score

    # Adjust score based on candle properties
    if is_bullish:
        score += 10
        if body_to_wick_ratio > 2:  # Strong bullish candle
            score += 20
        if last_candle['close'] > prev_candle['high']:  # Breakout candle
            score += 20
    else:
        score -= 10
        if body_to_wick_ratio > 2:  # Strong bearish candle
            score -= 20
        if last_candle['close'] < prev_candle['low']:  # Breakdown candle
            score -= 20

    return {
        'score': max(0, min(100, score)),
        'is_bullish': is_bullish,
        'strength': body_to_wick_ratio,
        'breakout': last_candle['close'] > prev_candle['high'] if is_bullish else last_candle['close'] < prev_candle['low']
    }


def calculate_volume_weighted_momentum(data: pd.DataFrame) -> dict:
    """Calculate volume-weighted momentum"""
    # Calculate volume-weighted price changes
    vol_weighted_change = (data['close'] - data['open']) * data['volume']

    return {
        'short_term': vol_weighted_change.tail(5).mean(),
        'medium_term': vol_weighted_change.mean(),
        'trend': 'bullish' if vol_weighted_change.tail(5).mean() > 0 else 'bearish',
        'strength': abs(vol_weighted_change.tail(5).mean()) / data['volume'].tail(5).mean()
    }


def calculate_trend_strength(data: pd.DataFrame) -> dict:
    """Calculate trend strength using price action"""
    closes = data['close']
    highs = data['high']
    lows = data['low']

    # Calculate basic trend properties
    higher_highs = sum(highs.diff() > 0)
    higher_lows = sum(lows.diff() > 0)

    # Calculate trend score
    score = ((higher_highs + higher_lows) / (2 * len(data))) * 100

    return {
        'score': score,
        'higher_highs': higher_highs,
        'higher_lows': higher_lows,
        'trend': 'bullish' if score > 50 else 'bearish',
        'strength': abs(50 - score)  # How far from neutral
    }


def calculate_momentum_score(price_change: float, pattern_score: float, volume_score: float) -> float:
    """Calculate final momentum score combining different factors"""
    # Weights for different components
    weights = {
        'price_change': 0.4,
        'pattern': 0.3,
        'volume': 0.3
    }

    # Normalize price change to 0-100 scale
    price_score = max(0, min(100, 50 + price_change))

    # Calculate weighted average
    final_score = (
        price_score * weights['price_change'] +
        pattern_score * weights['pattern'] +
        volume_score * weights['volume']
    )

    return max(0, min(100, final_score))


def calculate_price_momentum(data: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Calculate comprehensive price momentum indicators

    Parameters:
    -----------
    data: DataFrame with OHLCV data
    lookback: Number of periods for momentum calculation

    Returns:
    --------
    dict: Dictionary containing various momentum indicators
    """
    # Make sure we have enough data
    if len(data) < lookback:
        raise ValueError(
            f"Not enough data for {lookback} period momentum calculation")

    # 1. Price Change Momentum
    momentum = {
        'short_term': {
            'pct_change': data['close'].pct_change(5).iloc[-1] * 100,
            'direction': 1 if data['close'].pct_change(5).iloc[-1] > 0 else -1
        },
        'medium_term': {
            'pct_change': data['close'].pct_change(lookback).iloc[-1] * 100,
            'direction': 1 if data['close'].pct_change(lookback).iloc[-1] > 0 else -1
        }
    }

    # 2. Rate of Change (ROC)
    roc = {
        'short_term': (data['close'].iloc[-1] - data['close'].iloc[-4]) / data['close'].iloc[-4] * 100,
        'medium_term': (data['close'].iloc[-1] - data['close'].iloc[-11]) / data['close'].iloc[-11] * 100
    }

    return {
        'momentum': momentum,
        'roc': roc,
        'candle_momentum': analyze_candle_momentum(data.tail(3)),
        'volume_momentum': calculate_volume_weighted_momentum(data.tail(lookback)),
        'trend_strength': calculate_trend_strength(data.tail(lookback))
    }
