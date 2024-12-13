from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from smartmoneyconcepts import smc


def get_binance_data(symbol='BTCUSDT', interval='15m', lookback='1 day ago UTC'):
    # Initialize Binance client - no API keys needed for public data
    client = Client()

    # Get historical klines/candlestick data
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=lookback
    )

    # Create DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Keep only the OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def main():
    # Get 15m data
    df_15m = get_binance_data(interval='15m', lookback='1 day ago UTC')
    print("\n15m BTC Data:")
    print(df_15m.head())

    # Get 1h data
    df_1h = get_binance_data(interval='1h', lookback='1 day ago UTC')
    print("\n1h BTC Data:")
    print(df_1h.head())

    # Test SMC functions
    print("\nTesting SMC functions on 15m data:")

    # Get swing highs and lows
    swing_data = smc.swing_highs_lows(df_15m)
    print("\nSwing Highs and Lows:")
    print(swing_data[swing_data['HighLow'].notna()].head())

    # Get Fair Value Gaps
    fvg_data = smc.fvg(df_15m)
    print("\nFair Value Gaps:")
    print(fvg_data[fvg_data['FVG'].notna()].head())

    # Get Order Blocks
    ob_data = smc.ob(df_15m, swing_data)
    print("\nOrder Blocks:")
    print(ob_data.head())
    print(ob_data[ob_data['OB'].notna()].head())

    # Get Previous High/Low
    prev_hl = smc.previous_high_low(df_15m)
    print("\nPrevious High/Low:")
    print(prev_hl.head())


if __name__ == "__main__":
    main()
