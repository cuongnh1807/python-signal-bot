import pandas as pd


def analyze_candle_momentum(candles: pd.DataFrame, n_candles: int = 5) -> dict:
    if len(candles) < n_candles:
        raise ValueError(f"Need at least {n_candles} candles")
    recent_candles = candles.tail(n_candles)
    last_candle = recent_candles.iloc[-1]
    prev_candle = recent_candles.iloc[-2]

    body_size = abs(last_candle['close'] - last_candle['open'])
    upper_wick = last_candle['high'] - \
        max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'],
                     last_candle['close']) - last_candle['low']
    is_bullish = last_candle['close'] > last_candle['open']
    body_to_wick_ratio = body_size / (upper_wick + lower_wick + 0.0001)

    score = 50
    if is_bullish:
        score += 10
        if body_to_wick_ratio > 2:
            score += 20
        if last_candle['close'] > prev_candle['high']:
            score += 20
    else:
        score -= 10
        if body_to_wick_ratio > 2:
            score -= 20
        if last_candle['close'] < prev_candle['low']:
            score -= 20

    return {
        'score': max(0, min(100, score)),
        'is_bullish': is_bullish,
        'strength': body_to_wick_ratio,
        'breakout': last_candle['close'] > prev_candle['high'] if is_bullish else last_candle['close'] < prev_candle['low']
    }


def calculate_volume_weighted_momentum(data: pd.DataFrame, short_term_period: int = 5) -> dict:
    vol_weighted_change = (data['close'] - data['open']) * data['volume']
    short_term = vol_weighted_change.tail(short_term_period).mean()
    medium_term = vol_weighted_change.mean()
    trend = 'bullish' if short_term > 0 else 'bearish'
    strength = abs(short_term) / \
        (data['volume'].tail(short_term_period).mean() + 0.0001)
    return {
        'short_term': short_term,
        'medium_term': medium_term,
        'trend': trend,
        'strength': strength
    }


def calculate_trend_strength(data: pd.DataFrame) -> dict:
    highs = data['high']
    lows = data['low']
    higher_highs = sum(highs.diff() > 0)
    higher_lows = sum(lows.diff() > 0)
    score = ((higher_highs + higher_lows) / (2 * len(data))) * 100
    return {
        'score': score,
        'higher_highs': higher_highs,
        'higher_lows': higher_lows,
        'trend': 'bullish' if score > 50 else 'bearish',
        'strength': abs(50 - score)
    }


def calculate_momentum_score(price_change: float, pattern_score: float, volume_score: float) -> float:
    weights = {'price_change': 0.4, 'pattern': 0.3, 'volume': 0.3}
    price_score = max(0, min(100, 50 + price_change))
    final_score = (
        price_score * weights['price_change'] +
        max(0, min(100, pattern_score)) * weights['pattern'] +
        max(0, min(100, volume_score)) * weights['volume']
    )
    return max(0, min(100, final_score))


def calculate_price_momentum(data: pd.DataFrame, lookback: int = 20, short_term_period: int = 10) -> dict:
    if len(data) < lookback:
        raise ValueError(
            f"Not enough data for {lookback} period momentum calculation")

    momentum = {
        'short_term': {
            'pct_change': data['close'].pct_change(short_term_period).iloc[-1] * 100,
            'direction': 1 if data['close'].pct_change(short_term_period).iloc[-1] > 0 else -1
        },
        'medium_term': {
            'pct_change': data['close'].pct_change(lookback).iloc[-1] * 100,
            'direction': 1 if data['close'].pct_change(lookback).iloc[-1] > 0 else -1
        }
    }

    roc = {
        'short_term': (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5] * 100,
        'medium_term': (data['close'].iloc[-1] - data['close'].iloc[-lookback]) / data['close'].iloc[-lookback] * 100
    }

    candle_momentum = analyze_candle_momentum(data.tail(3))
    volume_momentum = calculate_volume_weighted_momentum(data.tail(lookback))
    trend_strength = calculate_trend_strength(data.tail(lookback))

    final_score = calculate_momentum_score(
        momentum['short_term']['pct_change'],
        candle_momentum['score'],
        volume_momentum['strength'] * 100
    )

    return {
        'momentum': momentum,
        'roc': roc,
        'candle_momentum': candle_momentum,
        'volume_momentum': volume_momentum,
        'trend_strength': trend_strength,
        'final_momentum_score': final_score
    }


def adjust_precision(value, precision):
    return round(value, precision)
