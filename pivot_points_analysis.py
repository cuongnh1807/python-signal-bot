import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from binance.client import Client
from indicators.pivot_standards import pivot_points


class PivotPointsAnalyzer:
    def __init__(self):
        self.client = Client("", "")

    def get_historical_data(self, symbol='BTCUSDT', interval='15m', lookback_days=4):
        """Fetch historical klines/candlestick data"""
        start_time = datetime.now() - timedelta(days=lookback_days)
        start_ms = int(start_time.timestamp() * 1000)

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ms,
            limit=1000
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index('timestamp', inplace=True)
        return df

    def plot_pivot_points(self, df, pivot_type='Traditional', pivot_timeframe='Auto',
                          use_daily_based=True, show_labels=True, show_prices=True,
                          max_historical_pivots=10):
        """Plot pivot points with full TradingView-like styling"""

        # Calculate pivot points
        pivot_df = smc.pivot_points(
            df,
            pivot_type=pivot_type,
            pivot_timeframe=pivot_timeframe,
            use_daily_based=use_daily_based,
            show_labels=show_labels,
            show_prices=show_prices,
            max_historical_pivots=max_historical_pivots
        )

        # Create figure
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))

        # Color scheme for pivot levels
        colors = {
            'P': '#ffffff',   # White for pivot
            'R': '#ff4444',   # Red for resistance
            'S': '#44ff44'    # Green for support
        }

        # Add pivot lines and labels
        for col in pivot_df.columns:
            # Determine level type and color
            level_type = col[0]  # 'P', 'R' or 'S'
            color = colors.get(level_type, '#ffffff')

            # Adjust opacity for historical pivots
            period = col.split('_p')[-1] if '_p' in col else '0'
            opacity = 0.7 * (1 - float(period) * 0.2)

            # Add line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[pivot_df[col].iloc[0]] * len(df.index),
                name=col,
                line=dict(
                    color=color,
                    width=1,
                    dash='dash'
                ),
                opacity=max(opacity, 0.2)
            ))

            # Add label if enabled
            if show_labels:
                label_text = f"{col}"
                if show_prices:
                    label_text += f" ({pivot_df[col].iloc[0]:.2f})"

                fig.add_annotation(
                    x=df.index[-1],
                    y=pivot_df[col].iloc[0],
                    text=label_text,
                    showarrow=False,
                    xanchor='left',
                    font=dict(
                        color=color,
                        size=10
                    )
                )

        # Update layout
        fig.update_layout(
            title=f"{pivot_type} Pivot Points ({pivot_timeframe})",
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig


def main():
    analyzer = PivotPointsAnalyzer()
    data = analyzer.get_historical_data(
        symbol='BTCUSDT',
        interval='15m',
        lookback_days=2
    )

    pivot_df = pivot_points(
        ohlc=data,
        pivot_type='Traditional',
        pivot_timeframe='Auto',
        use_daily_based=True,
        max_historical_pivots=15
    )

    # Group columns by period
    current_period = [col for col in pivot_df.columns if '_p' not in col]
    historical = [col for col in pivot_df.columns if '_p' in col]

    # Print current period
    print("\nCurrent Period:")
    print("-" * 20)
    for col in sorted(current_period):
        print(f"{col:3}: {pivot_df[col].iloc[0]:,.2f}")

    # Print historical periods
    for i in range(1, 10):
        period_cols = [col for col in historical if f'_p{i}' in col]
        if period_cols:
            print(f"\nPeriod -{i}:")
            print("-" * 20)
            for col in sorted(period_cols):
                base_name = col.split('_')[0]
                print(f"{base_name:3}: {pivot_df[col].iloc[0]:,.2f}")

    # Plot different types of pivot points

        # fig = analyzer.plot_pivot_points(
        #     df=data,
        #     pivot_type=method,
        #     pivot_timeframe='m15',
        #     use_daily_based=True,
        #     show_labels=True,
        #     show_prices=True,
        #     max_historical_pivots=15
        # )
        # fig.show()


if __name__ == "__main__":
    main()
