import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from binance.client import Client
from smartmoneyconcepts.smc import smc


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

    def plot_pivot_points(self, data: pd.DataFrame, method='traditional', timeframe='m15', levels=3):
        """
        Plot candlestick chart with pivot points

        Parameters:
        - method: Pivot point calculation method
            'traditional', 'fibonacci', 'woodie', 'classic', 'demark', 'camarilla'
        - timeframe: Time period for calculations
            'm15': 15 minutes
            '1d': Daily
            '1w': Weekly
            '1M': Monthly
            '3M': Quarterly
            '12M': Yearly
        - levels: Number of levels to show (1-5)
        """

        # Calculate pivot points
        pivot_data = smc.pivot_points(
            data, method=method, timeframe=timeframe, levels=levels)

        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='BTCUSDT'
        )])

        # Color schemes for different methods
        colors = {
            'traditional': {'r': ['red', 'orange', 'pink'], 's': ['green', 'lightgreen', 'lime']},
            'woodie': {'r': ['purple', 'magenta', 'violet'], 's': ['cyan', 'turquoise', 'aqua']},
            'camarilla': {'r': ['red', 'crimson', 'darkred', 'maroon'],
                          's': ['green', 'seagreen', 'darkgreen', 'olive']},
            'demark': {'r': ['red'], 's': ['green']},
            'fibonacci': {'r': ['gold', 'orange', 'red'], 's': ['lightgreen', 'green', 'darkgreen']}
        }

        # Add pivot line
        fig.add_trace(go.Scatter(
            x=pivot_data.index,
            y=pivot_data['pivot'],
            mode='lines',
            name='Pivot',
            line=dict(color='yellow', width=1)
        ))

        # Add resistance levels
        r_colors = colors[method.lower()]['r']
        max_levels = len(r_colors)
        for i in range(1, min(levels + 1, max_levels + 1)):
            if f'r{i}' in pivot_data.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_data.index,
                    y=pivot_data[f'r{i}'],
                    mode='lines',
                    name=f'R{i}',
                    line=dict(color=r_colors[i-1], width=1, dash='dash')
                ))

        # Add support levels
        s_colors = colors[method.lower()]['s']
        for i in range(1, min(levels + 1, max_levels + 1)):
            if f's{i}' in pivot_data.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_data.index,
                    y=pivot_data[f's{i}'],
                    mode='lines',
                    name=f'S{i}',
                    line=dict(color=s_colors[i-1], width=1, dash='dash')
                ))

        # Update layout
        fig.update_layout(
            title=f'BTCUSDT with {method.capitalize()} Pivot Points ({timeframe})',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis_rangeslider_visible=False
        )

        # Print current levels
        print(f"\nCurrent {method.capitalize()} Pivot Levels ({timeframe}):")
        print(f"Pivot: {pivot_data['pivot'].iloc[-1]:.2f}")

        for col in pivot_data.columns:
            if col != 'pivot':
                print(f"{col.upper()}: {pivot_data[col].iloc[-1]:.2f}")

        return fig


def main():
    analyzer = PivotPointsAnalyzer()
    data = analyzer.get_historical_data(
        symbol='BTCUSDT',
        interval='15m',
        lookback_days=1
    )

    # Plot different types of pivot points
    methods = ['traditional', 'fibonacci', 'woodie', 'demark', 'camarilla']

    for method in methods:
        print(f"\nAnalyzing {method.capitalize()} Pivot Points...")
        analyzer.plot_pivot_points(
            data=data,
            method=method,
            timeframe='m15',
            levels=3
        )
        # fig.show()


if __name__ == "__main__":
    main()
