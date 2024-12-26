import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from binance.client import Client
from indicators.rsi_divergence import find_rsi_divergences
from indicators.trap import detect_traps, plot_with_traps
from strategy import calculate_rsi


def fetch_btc_data(interval='15m', lookback_hours=24):
    """Fetch BTC/USDT data from Binance"""
    client = Client()
    start_time = int(
        (datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)

    klines = client.get_historical_klines(
        "BTCUSDT",
        interval,
        start_str=start_time
    )

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])

    df[['open', 'high', 'low', 'close', 'volume']] = df[[
        'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('time', inplace=True)

    return df


def plot_divergences(data: pd.DataFrame, divergences: dict) -> go.Figure:
    """Create interactive plot showing RSI divergences"""

    data['rsi'] = calculate_rsi(data)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'RSI')
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['rsi'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=2, col=1
    )

    # Plot divergences
    colors = {
        'regular_bullish': 'green',
        'hidden_bullish': 'lightgreen',
        'regular_bearish': 'red',
        'hidden_bearish': 'pink'
    }

    for div_type, point_pairs in divergences.items():
        if not point_pairs:
            continue

        color = colors[div_type]

        for prev_idx, curr_idx in point_pairs:
            # Vẽ đường kẻ nối trên biểu đồ giá
            if 'bullish' in div_type:
                price_values = data['low']
            else:
                price_values = data['high']

            fig.add_trace(
                go.Scatter(
                    x=[data.index[prev_idx], data.index[curr_idx]],
                    y=[price_values[prev_idx], price_values[curr_idx]],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=2,
                        dash='dot'
                    ),
                    name=f'{div_type.replace("_", " ").title()} Price Line'
                ),
                row=1, col=1
            )

            # Vẽ đường kẻ nối trên biểu đồ RSI
            fig.add_trace(
                go.Scatter(
                    x=[data.index[prev_idx], data.index[curr_idx]],
                    y=[data['rsi'][prev_idx], data['rsi'][curr_idx]],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=2,
                        dash='dot'
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        title='RSI Divergences',
        xaxis_title='Time',
        yaxis_title='Price',
        height=800
    )

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=2, col=1)

    return fig


def test_with_sample_data():
    """Test the divergence detection with real BTC data"""
    # Fetch data
    data = fetch_btc_data(lookback_hours=24*5)

    # Find divergences
    divergences = find_rsi_divergences(
        data,
        rsi_period=14,
        left_bars=5,
        right_bars=5,
        range_lower=5,
        range_upper=60
    )
    traps = detect_traps(data)
    print(traps)

    # Create and show plot
    fig = plot_divergences(data, divergences)
    fig = plot_with_traps(fig, data, traps)
    fig.show()

    # Print analysis
    print("\nDivergence Analysis:")
    print("===================")

    for div_type, point_pairs in divergences.items():
        if point_pairs:
            print(f"\n{div_type.replace('_', ' ').title()}:")
            for prev_idx, curr_idx in point_pairs:
                print(f"\nPrevious Point:")
                print(f"Time: {data.index[prev_idx]}")
                print(f"Price: {data['close'][prev_idx]:.2f}")
                print(f"RSI: {data['rsi'][prev_idx]:.1f}")

                print(f"\nCurrent Point:")
                print(f"Time: {data.index[curr_idx]}")
                print(f"Price: {data['close'][curr_idx]:.2f}")
                print(f"RSI: {data['rsi'][curr_idx]:.1f}")
                print("-" * 40)


if __name__ == "__main__":
    test_with_sample_data()
