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


def should_keep_ob(df, ob, current_index, use_should_keep_ob=True):
    if not use_should_keep_ob:
        return True

    ob_direction = ob['direction']

    pin_bar_signal = is_pin_bar(df.iloc[current_index])
    engulfing_signal = is_engulfing(df, current_index)

    price_action_score = 0
    if ob_direction == 1:  # Bullish
        price_action_score = 1 if (
            pin_bar_signal == 1 or engulfing_signal == 1) else 0
        if df.iloc[current_index]['low'] < ob['bottom']:
            price_action_score *= 2
    else:
        price_action_score = 1 if (
            pin_bar_signal == -1 or engulfing_signal == -1) else 0
        if df.iloc[current_index]['high'] > ob['top']:
            price_action_score *= 2

    macd = df['macd'].iloc[current_index]
    macd_hist = df['macd_hist'].iloc[current_index]
    macd_signal_line = df['macd_signal'].iloc[current_index]

    if ob_direction == 1:  # Bullish
        macd_signal = (
            (macd > macd_signal_line) and (
                macd_hist > df['macd_hist'].iloc[current_index-1])
        )
    else:  # Bearish
        macd_signal = (
            (macd < macd_signal_line) and (
                macd_hist < df['macd_hist'].iloc[current_index-1])

        )
    total_score = price_action_score + macd_signal
    if total_score >= 2:
        return True
    elif total_score == 1 and price_action_score == 1:
        return True
    return False


def should_keep_ob_flexible(df, ob, current_index, use_should_keep_ob=True):
    if not use_should_keep_ob:
        return True

    ob_direction = ob['direction']
    current_candle = df.iloc[current_index]
    prev_candle = df.iloc[current_index-1] if current_index > 0 else None

    # 1. Price Action Score (max 100)
    price_action_score = 0
    pin_bar = is_pin_bar(current_candle)
    engulfing = is_engulfing(
        df, current_index) if prev_candle is not None else 0

    if ob_direction == 1:  # Bullish
        if pin_bar == 1:
            price_action_score += 40  # Increased from 20
        if engulfing == 1:
            price_action_score += 40  # Increased from 20
        if current_candle['low'] <= ob['bottom']:
            price_action_score += 20
    else:  # Bearish
        if pin_bar == -1:
            price_action_score += 40  # Increased from 20
        if engulfing == -1:
            price_action_score += 40  # Increased from 20
        if current_candle['high'] >= ob['top']:
            price_action_score += 20

    # 2. Momentum Score (max 100)
    momentum_score = 0
    macd = current_candle['macd']
    macd_signal = current_candle['macd_signal']
    macd_hist = current_candle['macd_hist']
    prev_hist = df['macd_hist'].iloc[current_index -
                                     1] if current_index > 0 else 0

    if ob_direction == 1:  # Bullish
        if macd > macd_signal:
            momentum_score += 50  # Increased from 15
        if macd_hist > prev_hist:
            momentum_score += 50  # Increased from 15
    else:  # Bearish
        if macd < macd_signal:
            momentum_score += 50  # Increased from 15
        if macd_hist < prev_hist:
            momentum_score += 50  # Increased from 15

    # 3. Volume Score (max 100)
    volume_score = 0
    avg_volume = df['volume'].rolling(20).mean().iloc[current_index]

    volume_ratio = current_candle['volume'] / avg_volume
    if volume_ratio > 2.0:
        volume_score = 100
    elif volume_ratio > 1.5:
        volume_score = 75
    elif volume_ratio > 1.2:
        volume_score = 50
    elif volume_ratio > 1.0:
        volume_score = 25

    # Calculate final score (0-100 range)
    final_score = (
        price_action_score * 0.4 +
        momentum_score * 0.3 +
        volume_score * 0.3
    )
    print("final_score", final_score)

    # Return True if score >= 30 (more lenient threshold)
    return final_score >= 30
