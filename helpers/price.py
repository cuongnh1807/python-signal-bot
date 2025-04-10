import pandas as pd


def analyze_candle_momentum(candles: pd.DataFrame) -> dict:
    last_candle = candles.iloc[-1]
    prev_candle = candles.iloc[-2]

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


def calculate_price_momentum(data: pd.DataFrame, lookback: int = 20, short_term_period: int = 5) -> dict:
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


def merge_overlapping_order_blocks(obs, threshold):
    """
    Merge overlapping order blocks with enhanced handling of quality metrics.

    Parameters:
        obs (list): List of order blocks
        threshold (float): Overlap threshold for merging (0-1)

    Returns:
        list: Merged order blocks list
    """
    if not obs:
        return []

    # Sort by position
    sorted_obs = sorted(obs, key=lambda x: (x['top'] + x['bottom']) / 2)

    merged = []
    current = sorted_obs[0]

    for next_ob in sorted_obs[1:]:
        # Calculate overlap
        overlap_height = min(
            current['top'], next_ob['top']) - max(current['bottom'], next_ob['bottom'])
        total_height = max(current['top'], next_ob['top']) - \
            min(current['bottom'], next_ob['bottom'])

        # If overlap is significant, merge
        if overlap_height > 0 and overlap_height / total_height >= threshold:
            # Merge boundaries
            merged_ob = {
                'direction': current['direction'],
                'top': max(current['top'], next_ob['top']),
                'bottom': min(current['bottom'], next_ob['bottom']),
                'left_time': min(current['left_time'], next_ob['left_time']),
                'mitigated': current['mitigated'] and next_ob['mitigated'],
                'mitigated_time': max(current['mitigated_time'], next_ob['mitigated_time']) if current['mitigated'] and next_ob['mitigated'] else None,
                'index': min(current['index'], next_ob['index']),
                'strength': max(current['strength'], next_ob['strength']),
                'atr': (current['atr'] + next_ob['atr']) / 2,
                'volume': max(current['volume'], next_ob['volume'])
            }

            # Calculate average price in merged zone
            merged_ob['avg'] = (merged_ob['top'] + merged_ob['bottom']) / 2
            merged_ob['height'] = merged_ob['top'] - merged_ob['bottom']

            # Enhanced handling of quality metrics from should_keep_ob
            if 'setup_quality' in current and 'setup_quality' in next_ob:
                # Base quality is the max of both blocks
                base_quality = max(current.get(
                    'setup_quality', 0), next_ob.get('setup_quality', 0))
                base_threshold = max(current.get(
                    'threshold', 0), next_ob.get('threshold', 0))

                # Add a bonus for overlapping blocks (confirms importance of zone)
                # Bonus scales with strength and is capped at 15 points
                strength_bonus = min(
                    15, ((current['strength'] + next_ob['strength'])/200) * 10)

                # Ensure we don't exceed 100
                merged_ob['setup_quality'] = min(
                    100, base_quality + strength_bonus)
                merged_ob['threshold'] = min(
                    100, base_threshold + strength_bonus)

                # Calculate improved entry quality
                if current.get('entry_quality') in ['Excellent', 'Good'] or next_ob.get('entry_quality') in ['Excellent', 'Good']:
                    # Upgrade entry quality for confirmed zones
                    if base_quality >= 75:
                        merged_ob['entry_quality'] = 'Excellent'
                    elif base_quality >= 60:
                        merged_ob['entry_quality'] = 'Good'
                    else:
                        merged_ob['entry_quality'] = max(current.get(
                            'entry_quality', 'Poor'), next_ob.get('entry_quality', 'Poor'))
                else:
                    merged_ob['entry_quality'] = max(current.get(
                        'entry_quality', 'Poor'), next_ob.get('entry_quality', 'Poor'))

                # Merge warnings, removing duplicates
                merged_ob['warnings'] = list(
                    set(current.get('warnings', []) + next_ob.get('warnings', [])))

                # Add a note about the merge for clarity
                merged_ob['warnings'].append(
                    f"Merged from {len(sorted_obs)} overlapping order blocks")

                # Calculate new score
                if 'score' in current and 'score' in next_ob:
                    merged_ob['score'] = max(current.get(
                        'score', 0), next_ob.get('score', 0)) + strength_bonus

            # Update current with merged
            current = merged_ob
        else:
            # No significant overlap, add current to merged list and move to next
            merged.append(current)
            current = next_ob

    # Add the final block
    merged.append(current)

    return merged
