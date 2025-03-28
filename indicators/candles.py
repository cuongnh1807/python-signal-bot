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
    if i > 0:
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        if (current['close'] > current['open'] and prev['close'] < prev['open'] and
                current['close'] > prev['open'] and current['open'] < prev['close']):
            return 1
        elif (current['close'] < current['open'] and prev['close'] > prev['open'] and
                current['close'] < prev['open'] and current['open'] > prev['close']):
            return -1
    return 0


def should_keep_ob(df, ob, current_index):
    ob_direction = ob['direction']
    valid_signals = 0
    for i in range(current_index, max(current_index - 2, -1), -1):
        pin_bar_signal = is_pin_bar(df.iloc[i])
        engulfing_signal = is_engulfing(df, i)

        if ob_direction == 1:
            if pin_bar_signal == 1 or engulfing_signal == 1:
                valid_signals += 1
        else:
            if pin_bar_signal == -1 or engulfing_signal == -1:
                valid_signals += 1

    price_action_signal = valid_signals >= 1

    macd = df['macd'].iloc[current_index]
    macd_hist = df['macd_hist'].iloc[current_index]
    macd_signal_line = df['macd_signal'].iloc[current_index]

    if ob_direction == 1:  # Bullish
        macd_signal = (
            (macd < 0 and macd_hist > 0) or
            (macd > macd_signal_line and macd < 0)
        )
    else:  # Bearish
        macd_signal = (
            (macd > 0 and macd_hist < 0) or
            (macd < macd_signal_line and macd > 0)
        )

    signals = [price_action_signal, macd_signal]
    return sum(signals) >= 1
