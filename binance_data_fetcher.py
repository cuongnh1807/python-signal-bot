from dotenv import load_dotenv
import os
from binance.client import Client
import pandas as pd
from datetime import datetime
from time_synchronizer import get_time_synchronizer, initialize_time_sync
import logging

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    def __init__(self, client=None, time_synchronizer=None):
        # Load environment variables if needed
        if not client:
            load_dotenv()
            self.client = Client(os.getenv(
                'BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

        # Initialize client with API credentials
        self.client = client

        # Use provided time synchronizer or get global one
        if time_synchronizer:
            self.time_sync = time_synchronizer
        else:
            try:
                self.time_sync = get_time_synchronizer()
            except RuntimeError:
                # Initialize if not already done
                self.time_sync = initialize_time_sync(
                    os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

    def get_historical_klines(self, symbol: str, interval: str, start_time: datetime, limit: int = 1000) -> pd.DataFrame:
        """Fetch historical klines/candlestick data with proper time synchronization"""
        start_ms = int(start_time.timestamp() * 1000)
        self.time_sync.sync_time()

        try:
            # Gọi API không có tham số timestamp
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_ms,
                limit=limit
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
                raise ValueError(
                    f"Missing required columns: {missing_columns}")

            return df

        except Exception as e:
            logger.error(f"Error fetching historical klines: {str(e)}")
            # Nếu có lỗi, thử đồng bộ hóa thời gian và thử lại một lần nữa
            self.time_sync.sync_time()

            # Thử lại sau khi đồng bộ hóa
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_ms,
                limit=limit
            )

            # Xử lý dữ liệu như trên
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
                raise ValueError(
                    f"Missing required columns: {missing_columns}")

            return df
