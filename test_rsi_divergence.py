import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from binance.client import Client
from strategy import detect_rsi_divergence


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

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


def plot_divergences(data: pd.DataFrame, divergences: dict) -> go.Figure:
    """Create interactive plot showing RSI divergences with improved visualization"""

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
    for div_type in ['bullish', 'bearish']:
        color = 'green' if div_type == 'bullish' else 'red'

        for div in divergences[div_type]:
            start = div['points']['start']
            end = div['points']['end']

            # Add price lines
            fig.add_trace(
                go.Scatter(
                    x=[start['time'], end['time']],
                    y=[start['price'], end['price']],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=2,
                        dash='dot'
                    ),
                    name=f'{div_type.capitalize()} Divergence'
                ),
                row=1, col=1
            )

            # Add RSI lines
            fig.add_trace(
                go.Scatter(
                    x=[start['time'], end['time']],
                    y=[start['rsi'], end['rsi']],
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

            # Add strength annotation
            fig.add_annotation(
                x=end['time'],
                y=end['price'],
                text=f"Strength: {div['strength']:.1f}",
                showarrow=True,
                arrowhead=1,
                row=1, col=1
            )

    # Update layout
    fig.update_layout(
        title='RSI Divergences',
        xaxis_title='Time',
        yaxis_title='Price',
        height=800
    )

    return fig


def test_with_sample_data():
    """Test the divergence detection with real BTC data"""
    # Fetch 48 hours of 15m BTC data
    data = fetch_btc_data(lookback_hours=24*7)
    divergences = detect_rsi_divergence(data, lookback=400)

    # Create and show plot
    fig = plot_divergences(data, divergences)
    fig.show()

    # Print analysis
    print("\nDivergence Analysis:")
    print("===================")

    for div_type in ['bullish', 'bearish']:
        if divergences[div_type]:  # If list is not empty
            print(f"\n{div_type.capitalize()} Divergences:")
            for i, div in enumerate(divergences[div_type], 1):
                print(f"\nDivergence #{i}:")
                print(f"Strength: {div['strength']:.1f}")
                print(
                    f"Time Range: {div['points']['start']['time']} -> {div['points']['end']['time']}")
                print(
                    f"Price: {div['points']['start']['price']:.2f} -> {div['points']['end']['price']:.2f}")
                print(
                    f"RSI: {div['points']['start']['rsi']:.1f} -> {div['points']['end']['rsi']:.1f}")


if __name__ == "__main__":
    test_with_sample_data()
