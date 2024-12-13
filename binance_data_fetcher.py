from dotenv import load_dotenv
import os
from binance.client import Client
import pandas as pd
from datetime import datetime


class BinanceDataFetcher:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Get API credentials from environment variables
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')

        # Initialize client with API credentials
        self.client = Client(api_key, api_secret)

    def get_historical_klines(self, symbol: str, interval: str, start_time: datetime) -> pd.DataFrame:
        """Fetch historical klines/candlestick data"""
        start_ms = int(start_time.timestamp() * 1000)

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ms,
            limit=1000
        )

        # Ensure column names are lowercase
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index('timestamp', inplace=True)

        # Verify the DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df
