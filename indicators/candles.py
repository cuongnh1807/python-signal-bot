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
