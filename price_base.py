import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta


class PriceBaseStrategy:
    def __init__(self, price_base_window=96, price_base_offset=50, volume_base_multiplier=1.5):
        # Initialize Binance client
        self.client = Client("", "")  # Empty strings for public data only
        self.price_base_window = price_base_window
        self.price_base_offset = price_base_offset
        self.volume_base_multiplier = volume_base_multiplier

    def get_binance_data(self, symbol='BTCUSDT', interval='15m', lookback='30 days ago UTC'):
        """
        Fetch historical data from Binance
        interval options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        try:
            # Get historical klines/candlestick data
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=lookback
            )

            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)

            # Convert string values to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def find_setups(self, data):
        # Calculate base values
        data = data.copy()
        data['price_base'] = data['close'].rolling(
            self.price_base_window).mean()
        data['volume_base'] = data['volume'].rolling(
            self.price_base_window).mean()

        # Initialize columns for setups
        data['long_setup'] = False
        data['short_setup'] = False

        # Find breakout setups
        breakout_long = (data['close'] > data['price_base'] + self.price_base_offset) & \
            (data['volume'] > data['volume_base']
             * self.volume_base_multiplier)

        breakout_short = (data['close'] < data['price_base'] - self.price_base_offset) & \
            (data['volume'] > data['volume_base']
             * self.volume_base_multiplier)

        # Find reversion setups
        reversion_long = (data['close'] < data['price_base'] - self.price_base_offset) & \
            (data['volume'] < data['volume_base'])

        reversion_short = (data['close'] > data['price_base'] + self.price_base_offset) & \
            (data['volume'] < data['volume_base'])

        # Combine setups
        data.loc[breakout_long | reversion_long, 'long_setup'] = True
        data.loc[breakout_short | reversion_short, 'short_setup'] = True

        return data

    def plot_setups(self, data):
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['close'], label='Price', color='blue')
        plt.plot(data.index, data['price_base'],
                 label='Price Base', color='orange', linestyle='--')

        # Plot long setups
        plt.scatter(data.index[data['long_setup']],
                    data.loc[data['long_setup'], 'close'],
                    label='Long Setup', color='green', marker='^')

        # Plot short setups
        plt.scatter(data.index[data['short_setup']],
                    data.loc[data['short_setup'], 'close'],
                    label='Short Setup', color='red', marker='v')

        plt.legend()
        plt.title("BTC Price Base Strategy Setups")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.show()


def print_latest_setups(data, lookback_periods=5):
    """
    Print latest setup details
    lookback_periods: Number of recent periods to check for setups
    """
    latest_data = data.tail(lookback_periods)

    print("\n=== Latest Trading Setups ===")
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # Find most recent setups
    latest_long = latest_data[latest_data['long_setup']].tail(1)
    latest_short = latest_data[latest_data['short_setup']].tail(1)

    if not latest_long.empty:
        print("\n🟢 LONG SETUP DETECTED:")
        print(f"Time: {latest_long.index[0]}")
        print(f"Price: ${latest_long['close'].iloc[0]:,.2f}")
        print(f"Price Base: ${latest_long['price_base'].iloc[0]:,.2f}")
        print(f"Volume: {latest_long['volume'].iloc[0]:,.2f}")
        print(f"Volume Base: {latest_long['volume_base'].iloc[0]:,.2f}")

    if not latest_short.empty:
        print("\n🔴 SHORT SETUP DETECTED:")
        print(f"Time: {latest_short.index[0]}")
        print(f"Price: ${latest_short['close'].iloc[0]:,.2f}")
        print(f"Price Base: ${latest_short['price_base'].iloc[0]:,.2f}")
        print(f"Volume: {latest_short['volume'].iloc[0]:,.2f}")
        print(f"Volume Base: {latest_short['volume_base'].iloc[0]:,.2f}")

    if latest_long.empty and latest_short.empty:
        print("\nNo setups found in recent periods")


def calculate_position_size(balance, risk_percent, entry_price, stop_loss_price, leverage):
    """
    Calculate the position size based on risk management.

    Parameters:
    - balance: float, account balance in USDT
    - risk_percent: float, percentage of balance to risk per trade
    - entry_price: float, entry price in USDT
    - stop_loss_price: float, stop loss price in USDT
    - leverage: int, trading leverage

    Returns:
    - position_size: float, quantity to trade
    """
    risk_amount = balance * (risk_percent / 100)
    risk_per_coin = abs(entry_price - stop_loss_price)

    if risk_per_coin == 0:
        raise ValueError("Stop loss price cannot be the same as entry price.")

    position_size = (risk_amount / risk_per_coin) * leverage
    return position_size


# Usage example
if __name__ == "__main__":
    # Initialize strategy
    strategy = PriceBaseStrategy()

    # Fetch data from Binance
    data = strategy.get_binance_data(
        symbol='BTCUSDT',
        interval='15m',
        lookback='2 days ago UTC'
    )

    if data is not None:
        # Find setups
        results = strategy.find_setups(data)

        # Plot results
        strategy.plot_setups(results)

        # Print summary
        print("\nStrategy Setup Summary:")
        print(f"Total Long Setups: {results['long_setup'].sum()}")
        print(f"Total Short Setups: {results['short_setup'].sum()}")

        # Print latest setups with details
        print_latest_setups(results)
