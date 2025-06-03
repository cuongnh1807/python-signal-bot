import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests

# Binance imports
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.helpers import round_step_size

# Strategy imports
from futures_strategy import FuturesStrategy
from binance_data_fetcher import BinanceDataFetcher
from helpers.price import adjust_precision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Simplified Telegram notifications"""

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if enabled and bot_token and chat_id:
            self.send_message("🤖 Simple Trading Bot Started")

    def send_message(self, message: str):
        if not self.enabled or not self.bot_token or not self.chat_id:
            return

        try:
            response = requests.post(self.base_url, data={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            })
            if response.status_code != 200:
                logger.error(f"Telegram error: {response.text}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def notify_signal(self, symbol: str, side: str, entry_price: float,
                      stop_loss: float, take_profit: float, setup_quality: float):
        """Notify about new trading signal"""
        emoji = "🟢" if side == "LONG" else "🔴"
        message = (
            f"{emoji} <b>New {side} Signal</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Entry: <b>${entry_price:.4f}</b>\n"
            f"Stop Loss: <b>${stop_loss:.4f}</b>\n"
            f"Take Profit: <b>${take_profit:.4f}</b>\n"
            f"Quality: <b>{setup_quality:.1f}%</b>\n"
            f"Risk/Reward: <b>{abs(take_profit-entry_price)/abs(entry_price-stop_loss):.2f}</b>"
        )
        self.send_message(message)

    def notify_fill(self, symbol: str, side: str, price: float, quantity: float):
        """Notify about order fill"""
        message = (
            f"✅ <b>Order Filled</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: <b>{side}</b>\n"
            f"Price: <b>${price:.4f}</b>\n"
            f"Quantity: <b>{quantity:.6f}</b>"
        )
        self.send_message(message)

    def notify_close(self, symbol: str, side: str, entry: float, exit: float,
                     pnl: float, reason: str):
        """Notify about position close"""
        emoji = "🟢" if pnl > 0 else "🔴"
        message = (
            f"{emoji} <b>Position Closed</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: <b>{side}</b>\n"
            f"Entry: <b>${entry:.4f}</b>\n"
            f"Exit: <b>${exit:.4f}</b>\n"
            f"PnL: <b>${pnl:.2f}</b>\n"
            f"Reason: <b>{reason}</b>"
        )
        self.send_message(message)


class SimpleTradingBot:
    """Simplified trading bot for single or multiple tickers"""

    def __init__(self, symbols: List[str], config: Dict):
        """
        Initialize simple trading bot

        Parameters:
        - symbols: List of trading symbols ['BTCUSDT', 'ETHUSDT', ...]
        - config: Trading configuration
        """
        self.symbols = symbols
        self.config = config
        self.running = False

        # Initialize Binance client
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )

        # Initialize components
        self.data_fetcher = BinanceDataFetcher(client=self.client)
        self.telegram = TelegramNotifier(
            bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
            enabled=config.get('telegram_enabled', True)
        )

        # Get symbol precision info
        self.symbol_info = self._get_symbol_info()

        # Initialize strategies for each symbol
        self.strategies = {}
        self.historical_data = {}
        self.current_prices = {}
        self.active_orders = {}  # {symbol: [orders]}
        self.open_positions = {}  # {symbol: [positions]}

        # Get initial capital
        self.initial_capital = self._get_usdt_balance()

        for symbol in symbols:
            self.strategies[symbol] = FuturesStrategy(
                initial_capital=self.initial_capital /
                len(symbols),  # Split capital
                max_risk_per_trade=config.get('max_risk_per_trade', 0.02),
                default_leverage=config.get('leverage', 10)
            )
            self.active_orders[symbol] = []
            self.open_positions[symbol] = []

        logger.info(
            f"Initialized SimpleTradingBot for {len(symbols)} symbols with ${self.initial_capital:.2f}")

    def _get_symbol_info(self) -> Dict:
        """Get symbol precision information"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = {}

            for symbol_data in exchange_info['symbols']:
                if symbol_data['symbol'] in self.symbols:
                    symbol_info[symbol_data['symbol']] = {
                        'pricePrecision': symbol_data['pricePrecision'],
                        'quantityPrecision': symbol_data['quantityPrecision'],
                        'tickSize': symbol_data['filters'][0]['tickSize']
                    }

            return symbol_info
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return {}

    def _get_usdt_balance(self) -> float:
        """Get USDT balance from futures account"""
        try:
            if self.config.get('test_mode', True):
                return 1000.0  # Test mode default

            account_info = self.client.futures_account_balance()
            for asset in account_info:
                if asset['asset'] == 'USDT':
                    return min(float(asset['availableBalance']), 1000.0)
            return 1000.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 1000.0

    def _fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data for symbol"""
        try:
            start_time = datetime.now() - timedelta(
                hours=self.config.get('lookback_hours', 48)
            )

            data = self.data_fetcher.get_historical_klines(
                symbol=symbol,
                interval=self.config.get('interval', '15m'),
                start_time=start_time
            )

            if not data.empty:
                self.current_prices[symbol] = data['close'].iloc[-1]

            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _analyze_symbol(self, symbol: str) -> List[Dict]:
        """Analyze single symbol and generate orders"""
        try:
            # Fetch latest data
            data = self._fetch_data(symbol)
            if data.empty or len(data) < 50:
                return []

            self.historical_data[symbol] = data

            # Run strategy analysis
            analysis = self.strategies[symbol].analyze_market(data)

            # Generate orders
            orders = self.strategies[symbol].generate_orders(
                analysis=analysis,
                min_setup_quality=self.config.get('min_setup_quality', 70.0),
                min_volume_ratio=self.config.get('min_volume_ratio', 2.0),
                respect_pressure=True,
                respect_warnings=True
            )

            # Add symbol to orders
            for order in orders:
                order['symbol'] = symbol

            return orders

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return []

    def _place_order(self, order: Dict) -> bool:
        """Place order on exchange"""
        try:
            symbol = order['symbol']

            # Check distance from current price
            current_price = self.current_prices.get(symbol, 0)
            if current_price == 0:
                return False

            distance_pct = abs(
                (order['entry_price'] / current_price) - 1) * 100
            max_distance = self.config.get('max_distance_pct', 3.0)

            if distance_pct > max_distance:
                logger.info(
                    f"Skip {symbol} order: {distance_pct:.2f}% away from current price")
                return False

            # Send notification
            self.telegram.notify_signal(
                symbol=symbol,
                side=order['side'],
                entry_price=order['entry_price'],
                stop_loss=order['stop_loss'],
                take_profit=order['take_profit']['tp1'],
                setup_quality=order.get('setup_quality', 0)
            )

            if self.config.get('test_mode', True):
                logger.info(
                    f"TEST MODE: {symbol} {order['side']} order at {order['entry_price']:.4f}")
                order['status'] = 'TEST'
                self.active_orders[symbol].append(order)
                return True

            # Set leverage
            self.client.futures_change_leverage(
                symbol=symbol,
                leverage=order['leverage']
            )

            # Calculate quantity
            position_size = order['position_size']
            quantity = position_size / order['entry_price']
            quantity = adjust_precision(
                quantity, self.symbol_info[symbol]['quantityPrecision'])

            if quantity <= 0:
                logger.warning(f"Invalid quantity for {symbol}: {quantity}")
                return False

            # Place limit order
            response = self.client.futures_create_order(
                symbol=symbol,
                side='BUY' if order['side'] == 'LONG' else 'SELL',
                type='LIMIT',
                quantity=quantity,
                timeInForce='GTC',
                price=round_step_size(
                    order['entry_price'],
                    float(self.symbol_info[symbol]['tickSize'])
                )
            )

            order['order_id'] = response['orderId']
            order['status'] = 'PENDING'
            order['quantity'] = quantity
            self.active_orders[symbol].append(order)

            logger.info(
                f"Placed {symbol} {order['side']} order: {response['orderId']}")
            return True

        except Exception as e:
            logger.error(
                f"Error placing order for {order.get('symbol', 'unknown')}: {e}")
            return False

    def _check_orders(self):
        """Check status of all active orders"""
        for symbol in self.symbols:
            try:
                if self.config.get('test_mode', True):
                    self._simulate_order_fills(symbol)
                else:
                    self._check_real_orders(symbol)
            except Exception as e:
                logger.error(f"Error checking orders for {symbol}: {e}")

    def _simulate_order_fills(self, symbol: str):
        """Simulate order fills in test mode"""
        current_price = self.current_prices.get(symbol, 0)
        if current_price == 0:
            return

        for order in self.active_orders[symbol][:]:
            if order['status'] != 'TEST':
                continue

            # Check if limit order should fill
            should_fill = False
            if order['side'] == 'LONG' and current_price <= order['entry_price']:
                should_fill = True
            elif order['side'] == 'SHORT' and current_price >= order['entry_price']:
                should_fill = True

            if should_fill:
                # Simulate fill
                order['status'] = 'FILLED'
                order['fill_price'] = order['entry_price']
                order['fill_time'] = datetime.now()

                # Move to positions
                self.open_positions[symbol].append(order)
                self.active_orders[symbol].remove(order)

                self.telegram.notify_fill(
                    symbol=symbol,
                    side=order['side'],
                    price=order['entry_price'],
                    quantity=order.get('quantity', 0)
                )

                logger.info(
                    f"TEST: {symbol} {order['side']} filled at {order['entry_price']:.4f}")

    def _check_real_orders(self, symbol: str):
        """Check real orders on exchange"""
        try:
            open_orders = self.client.futures_get_open_orders(symbol=symbol)
            open_order_ids = [str(order['orderId']) for order in open_orders]

            for order in self.active_orders[symbol][:]:
                if order.get('order_id') and str(order['order_id']) not in open_order_ids:
                    # Order no longer open - check if filled
                    order_info = self.client.futures_get_order(
                        symbol=symbol,
                        orderId=order['order_id']
                    )

                    if order_info['status'] == 'FILLED':
                        order['status'] = 'FILLED'
                        order['fill_price'] = float(order_info['avgPrice'])
                        order['fill_time'] = datetime.fromtimestamp(
                            order_info['updateTime'] / 1000)

                        # Place stop loss and take profit
                        self._place_exit_orders(order)

                        # Move to positions
                        self.open_positions[symbol].append(order)
                        self.active_orders[symbol].remove(order)

                        self.telegram.notify_fill(
                            symbol=symbol,
                            side=order['side'],
                            price=order['fill_price'],
                            quantity=order.get('quantity', 0)
                        )
                    else:
                        # Order cancelled/rejected
                        self.active_orders[symbol].remove(order)

        except Exception as e:
            logger.error(f"Error checking real orders for {symbol}: {e}")

    def _place_exit_orders(self, order: Dict):
        """Place stop loss and take profit orders"""
        try:
            symbol = order['symbol']
            quantity = order['quantity']

            # Place stop loss
            self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if order['side'] == 'LONG' else 'BUY',
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=round_step_size(
                    order['stop_loss'],
                    float(self.symbol_info[symbol]['tickSize'])
                ),
                closePosition=True
            )

            # Place take profit
            tp_price = order['take_profit']['tp1']
            self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if order['side'] == 'LONG' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=round_step_size(
                    tp_price,
                    float(self.symbol_info[symbol]['tickSize'])
                )
            )

            logger.info(f"Placed exit orders for {symbol}")

        except Exception as e:
            logger.error(f"Error placing exit orders: {e}")

    def _check_positions(self):
        """Check position status and simulate exits in test mode"""
        for symbol in self.symbols:
            current_price = self.current_prices.get(symbol, 0)
            if current_price == 0:
                continue

            for position in self.open_positions[symbol][:]:
                if position['status'] != 'FILLED':
                    continue

                # Check stop loss
                if ((position['side'] == 'LONG' and current_price <= position['stop_loss']) or
                        (position['side'] == 'SHORT' and current_price >= position['stop_loss'])):

                    self._close_position(position, current_price, 'STOP_LOSS')

                # Check take profit
                elif ((position['side'] == 'LONG' and current_price >= position['take_profit']['tp1']) or
                      (position['side'] == 'SHORT' and current_price <= position['take_profit']['tp1'])):

                    self._close_position(
                        position, current_price, 'TAKE_PROFIT')

    def _close_position(self, position: Dict, exit_price: float, reason: str):
        """Close position and calculate PnL"""
        try:
            symbol = position['symbol']
            entry_price = position['fill_price']
            position_size = position['position_size']

            # Calculate PnL
            if position['side'] == 'LONG':
                pnl = position_size * (exit_price - entry_price) / entry_price
            else:
                pnl = position_size * (entry_price - exit_price) / entry_price

            position['status'] = 'CLOSED'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['pnl'] = pnl
            position['exit_reason'] = reason

            # Remove from open positions
            self.open_positions[symbol].remove(position)

            # Send notification
            self.telegram.notify_close(
                symbol=symbol,
                side=position['side'],
                entry=entry_price,
                exit=exit_price,
                pnl=pnl,
                reason=reason
            )

            logger.info(
                f"{symbol} {position['side']} closed: {reason}, PnL: ${pnl:.2f}")

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def _trading_loop(self):
        """Main trading loop"""
        last_analysis = {}

        while self.running:
            try:
                current_time = datetime.now()

                # Analyze each symbol every 3 minutes
                for symbol in self.symbols:
                    if (symbol not in last_analysis or
                            (current_time - last_analysis[symbol]).total_seconds() >= 180):

                        logger.info(f"Analyzing {symbol}...")

                        # Generate new orders
                        orders = self._analyze_symbol(symbol)

                        # Place new orders
                        for order in orders:
                            if self._should_place_order(order):
                                self._place_order(order)

                        last_analysis[symbol] = current_time

                # Check existing orders and positions
                self._check_orders()
                self._check_positions()

                # Clean old orders (remove pending orders older than 1 hour)
                self._clean_old_orders()

                # Sleep for 30 seconds
                time.sleep(30)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)

    def _should_place_order(self, order: Dict) -> bool:
        """Check if order should be placed (avoid duplicates)"""
        symbol = order['symbol']

        # Check for similar existing orders
        for existing_order in self.active_orders[symbol]:
            if (existing_order['side'] == order['side'] and
                    abs(existing_order['entry_price'] - order['entry_price']) < 0.001):
                return False

        return True

    def _clean_old_orders(self):
        """Remove old pending orders"""
        current_time = datetime.now()

        for symbol in self.symbols:
            for order in self.active_orders[symbol][:]:
                if (hasattr(order, 'created_time') and
                        (current_time - order.get('created_time', current_time)).total_seconds() > 3600):

                    if not self.config.get('test_mode', True) and order.get('order_id'):
                        try:
                            self.client.futures_cancel_order(
                                symbol=symbol,
                                orderId=order['order_id']
                            )
                        except:
                            pass

                    self.active_orders[symbol].remove(order)
                    logger.info(f"Removed old order for {symbol}")

    def start(self):
        """Start the trading bot"""
        if self.running:
            return

        logger.info("Starting Simple Trading Bot...")
        self.running = True

        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()

        logger.info(f"Bot started for symbols: {', '.join(self.symbols)}")

        # Send notification
        self.telegram.send_message(
            f"🚀 <b>Simple Trading Started</b>\n\n"
            f"Symbols: {', '.join(self.symbols)}\n"
            f"Capital: ${self.initial_capital:.2f}\n"
            f"Mode: {'Test' if self.config.get('test_mode', True) else 'Live'}"
        )

    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Simple Trading Bot...")
        self.running = False

        # Cancel all pending orders in live mode
        if not self.config.get('test_mode', True):
            for symbol in self.symbols:
                try:
                    self.client.futures_cancel_all_open_orders(symbol=symbol)
                except:
                    pass

        self.telegram.send_message("🛑 <b>Simple Trading Stopped</b>")
        logger.info("Bot stopped")

    def get_status(self) -> Dict:
        """Get current bot status"""
        total_orders = sum(len(orders)
                           for orders in self.active_orders.values())
        total_positions = sum(len(positions)
                              for positions in self.open_positions.values())

        return {
            'running': self.running,
            'symbols': self.symbols,
            'total_active_orders': total_orders,
            'total_open_positions': total_positions,
            'capital': self.initial_capital,
            'test_mode': self.config.get('test_mode', True),
            'current_prices': self.current_prices
        }


def load_config() -> Dict:
    """Load configuration from environment and defaults"""
    return {
        'test_mode': os.getenv('TEST_MODE', 'true').lower() == 'true',
        'interval': os.getenv('TRADING_INTERVAL', '15m'),
        'max_risk_per_trade': float(os.getenv('MAX_RISK_PER_TRADE', '0.02')),
        'leverage': int(os.getenv('LEVERAGE', '10')),
        'min_setup_quality': float(os.getenv('MIN_SETUP_QUALITY', '70.0')),
        'min_volume_ratio': float(os.getenv('MIN_VOLUME_RATIO', '2.0')),
        'max_distance_pct': float(os.getenv('MAX_DISTANCE_PCT', '3.0')),
        'telegram_enabled': os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true',
        'lookback_hours': int(os.getenv('LOOKBACK_HOURS', '48'))
    }


def main():
    """Main function"""
    load_dotenv()

    # Configuration
    config = load_config()

    # Symbols to trade (can be configured via environment)
    symbols_str = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,ADAUSDT')
    symbols = [s.strip() for s in symbols_str.split(',')]

    # Create and start bot
    bot = SimpleTradingBot(symbols=symbols, config=config)

    try:
        bot.start()

        # Keep running
        while True:
            time.sleep(300)  # Check status every 5 minutes
            status = bot.get_status()
            logger.info(f"Status: {status['total_active_orders']} orders, "
                        f"{status['total_open_positions']} positions")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        bot.stop()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        bot.stop()


if __name__ == "__main__":
    main()
