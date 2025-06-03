import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests

# Binance imports
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.helpers import round_step_size

# Import flux orderblock algorithm
from indicators.flux_orderblock import detect_flux_order_blocks, OrderBlockInfo
from binance_data_fetcher import BinanceDataFetcher
from helpers.price import adjust_precision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flux_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram notifications for flux orderblock trading"""

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if enabled and bot_token and chat_id:
            self.send_message("🚀 <b>Flux OrderBlock Trading Bot Started</b>")

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

    def notify_orderblock_signal(self, symbol: str, ob: OrderBlockInfo, entry_price: float,
                                 stop_loss: float, take_profit: float, position_size: float):
        """Notify about new order block signal"""
        emoji = "🟢" if ob.ob_type == "Bull" else "🔴"
        side = "LONG" if ob.ob_type == "Bull" else "SHORT"

        # Calculate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        message = (
            f"{emoji} <b>FLUX ORDER BLOCK {side}</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Entry: <b>${entry_price:.4f}</b>\n"
            f"Stop Loss: <b>${stop_loss:.4f}</b>\n"
            f"Take Profit: <b>${take_profit:.4f}</b>\n"
            f"Position Size: <b>${position_size:.2f}</b>\n"
            f"Risk/Reward: <b>{rr_ratio:.2f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
        )

        # Add entry evaluation metrics if available
        if hasattr(ob, 'entry_score'):
            message += (
                f"📊 <b>Entry Analysis</b>\n"
                f"Score: <b>{ob.entry_score:.1f}/100</b>\n"
                f"Quality: <b>{ob.entry_quality}</b>\n"
                f"Risk Level: <b>{ob.risk_level}</b>\n"
            )

            if hasattr(ob, 'trend_confluence'):
                conf = ob.trend_confluence
                message += (
                    f"Trend: <b>{conf['primary_trend'].title()}</b>\n"
                    f"Supporting TFs: <b>{conf['supporting_timeframes']}/3</b>\n"
                )
                if conf['trend_change']:
                    message += "🔄 <b>Trend Change Detected</b>\n"

            if hasattr(ob, 'warnings') and ob.warnings:
                message += f"⚠️ <b>Warnings:</b> {', '.join(ob.warnings[:2])}\n"

        # Order block details
        message += (
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Order Block Details</b>\n"
            f"Range: <b>{ob.bottom:.4f} - {ob.top:.4f}</b>\n"
            f"Height: <b>{abs(ob.top - ob.bottom):.6f}</b>\n"
            f"Volume: <b>{ob.ob_volume:.0f}</b>\n"
            f"Status: <b>{'🔴 Broken' if ob.breaker else '🟢 Active'}</b>\n"
            f"Created: <b>{ob.start_time.strftime('%m-%d %H:%M')}</b>"
        )

        self.send_message(message)

    def notify_fill(self, symbol: str, side: str, price: float, quantity: float):
        """Notify about order fill"""
        emoji = "✅"
        message = (
            f"{emoji} <b>Order Filled</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: <b>{side}</b>\n"
            f"Price: <b>${price:.4f}</b>\n"
            f"Quantity: <b>{quantity:.6f}</b>"
        )
        self.send_message(message)

    def notify_close(self, symbol: str, side: str, entry: float, exit: float,
                     pnl: float, pnl_percent: float, reason: str):
        """Notify about position close"""
        emoji = "🟢" if pnl > 0 else "🔴"
        message = (
            f"{emoji} <b>Position Closed</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: <b>{side}</b>\n"
            f"Entry: <b>${entry:.4f}</b>\n"
            f"Exit: <b>${exit:.4f}</b>\n"
            f"PnL: <b>${pnl:.2f} ({pnl_percent:.2f}%)</b>\n"
            f"Reason: <b>{reason}</b>"
        )
        self.send_message(message)


class FluxOrderBlockStrategy:
    """Strategy class for Flux Order Block trading"""

    def __init__(self, symbol: str, config: Dict):
        self.symbol = symbol
        self.config = config
        self.last_orderblocks = []
        self.last_analysis_time = None

    def analyze_and_generate_orders(self, data: pd.DataFrame) -> List[Dict]:
        """
        Analyze market using Flux Order Blocks and generate trading orders

        Parameters:
        - data: OHLCV DataFrame

        Returns:
        - List of order dictionaries
        """
        try:
            # Detect flux order blocks with entry evaluation
            order_blocks = detect_flux_order_blocks(
                data,
                swing_length=self.config.get('swing_length', 10),
                max_atr_mult=self.config.get('max_atr_mult', 3.5),
                ob_end_method=self.config.get('mitigation_method', 'Wick'),
                bullish_ob_count=self.config.get('max_bullish_obs', 5),
                bearish_ob_count=self.config.get('max_bearish_obs', 5),
                use_entry_evaluation=self.config.get(
                    'use_entry_evaluation', True),
                entry_threshold=self.config.get('entry_threshold', 45)
            )

            if not order_blocks:
                logger.info(f"{self.symbol}: No quality order blocks found")
                return []

            # Filter only non-broken order blocks
            active_obs = [ob for ob in order_blocks if not ob.breaker]

            if not active_obs:
                logger.info(f"{self.symbol}: No active order blocks")
                return []

            # Generate orders from top quality order blocks
            orders = []
            current_price = data['close'].iloc[-1]

            for ob in active_obs[:self.config.get('max_orders_per_symbol', 2)]:
                order = self._create_order_from_orderblock(
                    ob, current_price, data)
                if order:
                    orders.append(order)

            # Store for next analysis
            self.last_orderblocks = order_blocks
            self.last_analysis_time = datetime.now()

            logger.info(
                f"{self.symbol}: Generated {len(orders)} orders from {len(active_obs)} active OBs")
            return orders

        except Exception as e:
            logger.error(f"Error analyzing {self.symbol}: {e}")
            return []

    def _create_order_from_orderblock(self, ob: OrderBlockInfo, current_price: float,
                                      data: pd.DataFrame) -> Optional[Dict]:
        """Create trading order from order block"""
        try:
            # Determine order direction and prices
            if ob.ob_type == "Bull":
                side = "LONG"
                entry_price = ob.bottom  # Enter at bottom of bullish OB
                stop_loss = entry_price * \
                    (1 - self.config.get('stop_loss_pct', 0.02))
                take_profit = entry_price * \
                    (1 + self.config.get('take_profit_pct', 0.04))
            else:  # Bear
                side = "SHORT"
                entry_price = ob.top  # Enter at top of bearish OB
                stop_loss = entry_price * \
                    (1 + self.config.get('stop_loss_pct', 0.02))
                take_profit = entry_price * \
                    (1 - self.config.get('take_profit_pct', 0.04))

            # Check distance from current price
            distance_pct = abs((entry_price / current_price) - 1) * 100
            max_distance = self.config.get('max_distance_pct', 5.0)

            if distance_pct > max_distance:
                logger.info(
                    f"{self.symbol}: OB too far ({distance_pct:.1f}%) - skipping")
                return None

            # Calculate position size based on risk
            risk_per_trade = self.config.get('risk_per_trade_pct', 2.0) / 100
            capital_per_symbol = self.config.get('capital_per_symbol', 500)

            risk_amount = capital_per_symbol * risk_per_trade
            price_risk = abs(entry_price - stop_loss)

            if price_risk <= 0:
                return None

            position_size = risk_amount / (price_risk / entry_price)

            # Limit position size
            max_position_size = capital_per_symbol * 0.2  # Max 20% of capital per trade
            position_size = min(position_size, max_position_size)

            order = {
                'symbol': self.symbol,
                'side': side,
                'type': 'LIMIT',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'leverage': self.config.get('leverage', 10),
                'order_block': ob,  # Store reference to OB
                'created_time': datetime.now(),
                'distance_pct': distance_pct
            }

            return order

        except Exception as e:
            logger.error(f"Error creating order from OB: {e}")
            return None


class FluxLiveTradingBot:
    """Live trading bot using Flux Order Block strategy"""

    def __init__(self, symbols: List[str], config: Dict):
        """
        Initialize Flux Order Block trading bot

        Parameters:
        - symbols: List of trading symbols
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
        self.current_prices = {}
        self.active_orders = {}  # {symbol: [orders]}
        self.open_positions = {}  # {symbol: [positions]}

        # Initialize strategies
        for symbol in symbols:
            self.strategies[symbol] = FluxOrderBlockStrategy(symbol, config)
            self.active_orders[symbol] = []
            self.open_positions[symbol] = []

        # Get initial capital
        self.initial_capital = self._get_usdt_balance()
        capital_per_symbol = self.initial_capital / len(symbols)
        self.config['capital_per_symbol'] = capital_per_symbol

        logger.info(
            f"Initialized FluxLiveTradingBot for {len(symbols)} symbols")
        logger.info(
            f"Total capital: ${self.initial_capital:.2f}, Per symbol: ${capital_per_symbol:.2f}")

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
                return self.config.get('test_capital', 1000.0)

            account_info = self.client.futures_account_balance()
            for asset in account_info:
                if asset['asset'] == 'USDT':
                    available = float(asset['availableBalance'])
                    return min(available, self.config.get('max_capital', 1000.0))
            return 1000.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 1000.0

    def _fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data for symbol"""
        try:
            start_time = datetime.now() - timedelta(
                hours=self.config.get('lookback_hours', 168)
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
        """Analyze single symbol using Flux Order Block strategy"""
        try:
            # Fetch latest data
            data = self._fetch_data(symbol)
            if data.empty or len(data) < 100:
                logger.warning(f"{symbol}: Insufficient data for analysis")
                return []

            # Run Flux Order Block analysis
            orders = self.strategies[symbol].analyze_and_generate_orders(data)

            # Add symbol and additional info to orders
            for order in orders:
                order['symbol'] = symbol
                order['current_price'] = self.current_prices[symbol]

            return orders

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return []

    def _place_order(self, order: Dict) -> bool:
        """Place order on exchange"""
        try:
            symbol = order['symbol']
            ob = order['order_block']

            # Send Telegram notification
            self.telegram.notify_orderblock_signal(
                symbol=symbol,
                ob=ob,
                entry_price=order['entry_price'],
                stop_loss=order['stop_loss'],
                take_profit=order['take_profit'],
                position_size=order['position_size']
            )

            if self.config.get('test_mode', True):
                logger.info(
                    f"TEST MODE: {symbol} {order['side']} order at {order['entry_price']:.4f}")
                order['status'] = 'TEST'
                order['order_id'] = f"TEST_{int(time.time())}"
                self.active_orders[symbol].append(order)
                return True

            # Set leverage
            self.client.futures_change_leverage(
                symbol=symbol,
                leverage=order['leverage']
            )

            # Calculate quantity
            quantity = order['position_size'] / order['entry_price']
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
            self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if order['side'] == 'LONG' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=round_step_size(
                    order['take_profit'],
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
                elif ((position['side'] == 'LONG' and current_price >= position['take_profit']) or
                      (position['side'] == 'SHORT' and current_price <= position['take_profit'])):

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

            pnl_percent = (pnl / position_size) * 100

            position['status'] = 'CLOSED'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['pnl'] = pnl
            position['pnl_percent'] = pnl_percent
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
                pnl_percent=pnl_percent,
                reason=reason
            )

            logger.info(
                f"{symbol} {position['side']} closed: {reason}, PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def _trading_loop(self):
        """Main trading loop"""
        last_analysis = {}
        analysis_interval = self.config.get(
            'analysis_interval_minutes', 15) * 60  # Convert to seconds

        while self.running:
            try:
                current_time = datetime.now()

                # Analyze each symbol
                for symbol in self.symbols:
                    if (symbol not in last_analysis or
                            (current_time - last_analysis[symbol]).total_seconds() >= analysis_interval):

                        logger.info(
                            f"Analyzing {symbol} for Flux Order Blocks...")

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

                # Clean old orders
                self._clean_old_orders()

                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval_seconds', 60))

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(120)  # Wait longer on error

    def _should_place_order(self, order: Dict) -> bool:
        """Check if order should be placed (avoid duplicates)"""
        symbol = order['symbol']

        # Check for similar existing orders
        for existing_order in self.active_orders[symbol]:
            if (existing_order['side'] == order['side'] and
                    abs(existing_order['entry_price'] - order['entry_price']) / order['entry_price'] < 0.005):  # 0.5%
                return False

        # Limit max orders per symbol
        max_orders = self.config.get('max_orders_per_symbol', 2)
        if len(self.active_orders[symbol]) >= max_orders:
            logger.info(f"{symbol}: Max orders reached ({max_orders})")
            return False

        return True

    def _clean_old_orders(self):
        """Remove old pending orders"""
        current_time = datetime.now()
        max_age_hours = self.config.get('max_order_age_hours', 24)

        for symbol in self.symbols:
            for order in self.active_orders[symbol][:]:
                age_hours = (current_time - order.get('created_time',
                             current_time)).total_seconds() / 3600

                if age_hours > max_age_hours:
                    if not self.config.get('test_mode', True) and order.get('order_id'):
                        try:
                            self.client.futures_cancel_order(
                                symbol=symbol, orderId=order['order_id'])
                        except:
                            pass

                    self.active_orders[symbol].remove(order)
                    logger.info(
                        f"Removed old order for {symbol} (age: {age_hours:.1f}h)")

    def start(self):
        """Start the trading bot"""
        if self.running:
            return

        logger.info("Starting Flux Order Block Trading Bot...")
        self.running = True

        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()

        logger.info(f"Bot started for symbols: {', '.join(self.symbols)}")

        # Send startup notification
        mode = "TEST" if self.config.get('test_mode', True) else "LIVE"
        self.telegram.send_message(
            f"🚀 <b>Flux OrderBlock Bot Started</b>\n\n"
            f"Mode: <b>{mode}</b>\n"
            f"Symbols: <b>{', '.join(self.symbols)}</b>\n"
            f"Capital: <b>${self.initial_capital:.2f}</b>\n"
            f"Entry Evaluation: <b>{'✅ Enabled' if self.config.get('use_entry_evaluation', True) else '❌ Disabled'}</b>\n"
            f"Analysis Interval: <b>{self.config.get('analysis_interval_minutes', 15)} min</b>"
        )

    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Flux Order Block Trading Bot...")
        self.running = False

        # Cancel all pending orders in live mode
        if not self.config.get('test_mode', True):
            for symbol in self.symbols:
                try:
                    self.client.futures_cancel_all_open_orders(symbol=symbol)
                except:
                    pass

        self.telegram.send_message("🛑 <b>Flux OrderBlock Bot Stopped</b>")
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
            'current_prices': self.current_prices,
            'capital_per_symbol': self.config.get('capital_per_symbol', 0)
        }


def load_config() -> Dict:
    """Load configuration from environment and defaults"""
    return {
        # Trading mode
        'test_mode': os.getenv('TEST_MODE', 'true').lower() == 'true',
        'test_capital': float(os.getenv('TEST_CAPITAL', '1000')),
        'max_capital': float(os.getenv('MAX_CAPITAL', '1000')),

        # Market data
        'interval': os.getenv('TRADING_INTERVAL', '15m'),
        'lookback_hours': int(os.getenv('LOOKBACK_HOURS', '168')),

        # Flux OrderBlock parameters
        'swing_length': int(os.getenv('SWING_LENGTH', '10')),
        'max_atr_mult': float(os.getenv('MAX_ATR_MULT', '3.5')),
        'mitigation_method': os.getenv('MITIGATION_METHOD', 'Wick'),
        'max_bullish_obs': int(os.getenv('MAX_BULLISH_OBS', '5')),
        'max_bearish_obs': int(os.getenv('MAX_BEARISH_OBS', '5')),

        # Entry evaluation
        'use_entry_evaluation': True,
        'entry_threshold': int(os.getenv('ENTRY_THRESHOLD', '45')),

        # Risk management
        'risk_per_trade_pct': float(os.getenv('RISK_PER_TRADE_PCT', '2.0')),
        'leverage': int(os.getenv('LEVERAGE', '10')),
        'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '2.0')) / 100,
        'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '4.0')) / 100,
        'max_distance_pct': float(os.getenv('MAX_DISTANCE_PCT', '5.0')),

        # Order management
        'max_orders_per_symbol': int(os.getenv('MAX_ORDERS_PER_SYMBOL', '2')),
        'max_order_age_hours': int(os.getenv('MAX_ORDER_AGE_HOURS', '24')),

        # Timing
        'analysis_interval_minutes': int(os.getenv('ANALYSIS_INTERVAL_MINUTES', '15')),
        'monitoring_interval_seconds': int(os.getenv('MONITORING_INTERVAL_SECONDS', '60')),

        # Telegram
        'telegram_enabled': os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'
    }


def main():
    """Main function"""
    load_dotenv()

    # Configuration
    config = load_config()

    # Symbols to trade
    symbols_str = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,ADAUSDT')
    symbols = [s.strip() for s in symbols_str.split(',')]

    # Create and start bot
    bot = FluxLiveTradingBot(symbols=symbols, config=config)

    try:
        bot.start()

        # Keep running and show status periodically
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
