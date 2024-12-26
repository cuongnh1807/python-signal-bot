def detect_traps(df, window=20, volume_threshold=0.7, price_threshold=0.02):
    """
    Detect bull and bear traps

    Parameters:
    - window: số nến để tính support/resistance
    - volume_threshold: ngưỡng volume so với trung bình
    - price_threshold: ngưỡng % giá phá vỡ tối thiểu
    """

    def calculate_support_resistance(data, window):
        """Tính các mức support/resistance dựa trên local min/max"""
        highs = []
        lows = []

        for i in range(window, len(data)-window):
            # Kiểm tra local high
            if data['high'][i] == max(data['high'][i-window:i+window]):
                highs.append((i, data['high'][i]))

            # Kiểm tra local low
            if data['low'][i] == min(data['low'][i-window:i+window]):
                lows.append((i, data['low'][i]))

        return highs, lows

    def is_breakout(price, level, threshold):
        """Kiểm tra breakout/breakdown"""
        return abs(price - level) / level > threshold

    def check_volume_condition(volume, avg_volume, threshold):
        """Kiểm tra điều kiện volume thấp"""
        return volume < avg_volume * threshold

    # Tính support/resistance
    highs, lows = calculate_support_resistance(df, window)

    # Tính volume trung bình
    avg_volume = df['volume'].rolling(window).mean()

    traps = {
        'bull_traps': [],
        'bear_traps': []
    }

    # Tìm Bull Traps
    for i, resistance in highs:
        if i + 5 >= len(df):  # Bỏ qua các điểm cuối chuỗi
            continue

        # Kiểm tra breakout với volume thấp
        if (is_breakout(df['high'][i+1], resistance, price_threshold) and
                check_volume_condition(df['volume'][i+1], avg_volume[i+1], volume_threshold)):

            # Kiểm tra đảo chiều trong 5 nến tiếp theo
            future_prices = df['low'][i+1:i+6]
            if min(future_prices) < resistance:
                traps['bull_traps'].append({
                    'index': i+1,
                    'price': df['high'][i+1],
                    'resistance': resistance,
                    'reversal_price': min(future_prices)
                })

    # Tìm Bear Traps
    for i, support in lows:
        if i + 5 >= len(df):
            continue

        # Kiểm tra breakdown với volume thấp
        if (is_breakout(df['low'][i+1], support, price_threshold) and
                check_volume_condition(df['volume'][i+1], avg_volume[i+1], volume_threshold)):

            # Kiểm tra đảo chiều trong 5 nến tiếp theo
            future_prices = df['high'][i+1:i+6]
            if max(future_prices) > support:
                traps['bear_traps'].append({
                    'index': i+1,
                    'price': df['low'][i+1],
                    'support': support,
                    'reversal_price': max(future_prices)
                })

    return traps


def plot_with_traps(fig, data, traps):

    # Vẽ bull traps
    for trap in traps['bull_traps']:
        fig.add_shape(
            type="line",
            x0=data.index[trap['index']-5],
            y0=trap['resistance'],
            x1=data.index[trap['index']+5],
            y1=trap['resistance'],
            line=dict(color="red", width=1, dash="dot"),
            row=1, col=1
        )

        fig.add_annotation(
            x=data.index[trap['index']],
            y=trap['price'],
            text="Bull Trap",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )

    # Vẽ bear traps
    for trap in traps['bear_traps']:
        fig.add_shape(
            type="line",
            x0=data.index[trap['index']-5],
            y0=trap['support'],
            x1=data.index[trap['index']+5],
            y1=trap['support'],
            line=dict(color="green", width=1, dash="dot"),
            row=1, col=1
        )

        fig.add_annotation(
            x=data.index[trap['index']],
            y=trap['price'],
            text="Bear Trap",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )

    return fig
