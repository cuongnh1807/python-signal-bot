import time
import logging
from binance.client import Client
from typing import Dict, Optional
import threading

logger = logging.getLogger(__name__)


class BinanceTimeSynchronizer:
    """
    Class to handle time synchronization with Binance servers
    to prevent timestamp-related errors
    """

    def __init__(self, api_key: str, api_secret: str, auto_sync: bool = True, sync_interval: int = 3600):
        """
        Initialize the time synchronizer

        Parameters:
        -----------
        api_key: Binance API key
        api_secret: Binance API secret
        auto_sync: Whether to automatically sync time periodically
        sync_interval: Time between automatic syncs in seconds (default: 1 hour)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret)
        self.time_offset = 0
        self.last_sync_time = 0
        self.sync_interval = sync_interval
        self.sync_lock = threading.Lock()
        self.auto_sync = auto_sync

        # Perform initial synchronization
        self.sync_time()

        # Start automatic synchronization if enabled
        if auto_sync:
            self._start_auto_sync()

    def sync_time(self) -> int:
        """
        Synchronize time with Binance server

        Returns:
        --------
        Current server timestamp in milliseconds
        """
        with self.sync_lock:
            try:
                # Get server time from Binance
                server_time = self.client.get_server_time()
                server_timestamp = server_time['serverTime']

                # Get local time
                local_timestamp = int(time.time() * 1000)

                # Calculate offset
                self.time_offset = server_timestamp - local_timestamp

                # Update last sync time
                self.last_sync_time = local_timestamp

                logger.info(
                    f"Time synchronized with Binance server. Offset: {self.time_offset}ms")
                return server_timestamp

            except Exception as e:
                logger.error(
                    f"Error synchronizing time with Binance server: {str(e)}")
                # Return local time if synchronization fails
                return int(time.time() * 1000)

    def get_server_time(self) -> int:
        """
        Get current server time with offset applied

        Returns:
        --------
        Current estimated server time in milliseconds
        """
        # Check if we need to resync
        current_time = int(time.time() * 1000)
        if current_time - self.last_sync_time > self.sync_interval:
            self.sync_time()

        # Return adjusted time
        return current_time + self.time_offset

    def get_timestamp_with_recvwindow(self, recv_window: int = 5000) -> Dict[str, int]:
        """
        Get timestamp and recvWindow parameters for API requests

        Parameters:
        -----------
        recv_window: Receive window in milliseconds (default: 5000)

        Returns:
        --------
        Dictionary with timestamp and recvWindow parameters
        """
        return {
            'timestamp': self.get_server_time(),
            'recvWindow': recv_window
        }

    def _start_auto_sync(self):
        """Start automatic time synchronization thread"""
        def sync_thread():
            while self.auto_sync:
                # Sleep for the sync interval
                time.sleep(self.sync_interval)

                # Sync time
                try:
                    self.sync_time()
                except Exception as e:
                    logger.error(f"Error in automatic time sync: {str(e)}")

        # Start thread
        thread = threading.Thread(target=sync_thread, daemon=True)
        thread.start()
        logger.info(
            f"Automatic time synchronization started (interval: {self.sync_interval}s)")

    def stop_auto_sync(self):
        """Stop automatic time synchronization"""
        self.auto_sync = False
        logger.info("Automatic time synchronization stopped")


# Singleton instance for global use
_synchronizer: Optional[BinanceTimeSynchronizer] = None


def initialize_time_sync(api_key: str, api_secret: str, auto_sync: bool = True, sync_interval: int = 3600):
    """
    Initialize the global time synchronizer

    Parameters:
    -----------
    api_key: Binance API key
    api_secret: Binance API secret
    auto_sync: Whether to automatically sync time periodically
    sync_interval: Time between automatic syncs in seconds
    """
    global _synchronizer
    if _synchronizer is None:
        _synchronizer = BinanceTimeSynchronizer(
            api_key, api_secret, auto_sync, sync_interval)
    return _synchronizer


def get_time_synchronizer() -> BinanceTimeSynchronizer:
    """
    Get the global time synchronizer instance

    Returns:
    --------
    BinanceTimeSynchronizer instance

    Raises:
    -------
    RuntimeError if the synchronizer has not been initialized
    """
    global _synchronizer
    if _synchronizer is None:
        raise RuntimeError(
            "Time synchronizer has not been initialized. Call initialize_time_sync first.")
    return _synchronizer
