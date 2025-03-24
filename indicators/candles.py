def is_pin_bar(row):
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - max(row['open'], row['close'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    if body < 0.1 * (row['high'] - row['low']):
        if upper_shadow > 2 * body or lower_shadow > 2 * body:
            return True
    return False


def is_engulfing(df, i):
    if i > 0:
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        if (current['close'] > current['open'] and prev['close'] < prev['open'] and
                current['close'] > prev['open'] and current['open'] < prev['close']):
            return 'bullish'
        elif (current['close'] < current['open'] and prev['close'] > prev['open'] and
                current['close'] < prev['open'] and current['open'] > prev['close']):
            return 'bearish'
    return None


def should_keep_ob(df, ob, current_index):
    ob_direction = ob['direction']
    pin_bar_signal = is_pin_bar(df.iloc[current_index])
    engulfing_signal = is_engulfing(df, current_index)
    price_action_signal = pin_bar_signal or (engulfing_signal is not None)

    # 4. MACD Signal
    macd = df['macd'].iloc[current_index]
    macd_hist = df['macd_hist'].iloc[current_index]
    macd_signal_line = df['macd_signal'].iloc[current_index]

    if ob_direction == 'bullish':
        macd_signal = (macd < 0 and macd_hist > 0) or (
            macd > macd_signal_line and macd < 0)
    else:  # bearish
        macd_signal = (macd > 0 and macd_hist < 0) or (
            macd < macd_signal_line and macd > 0)
    signals = [price_action_signal, macd_signal]
    return sum(signals) >= 1
