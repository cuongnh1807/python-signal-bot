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

# Import both orderblock algorithms
from indicators.flux_orderblock import detect_flux_order_blocks, OrderBlockInfo, evaluate_flux_entry
from indicators.order_breaker_blocks import detect_order_breaker_blocks
from binance_data_fetcher import BinanceDataFetcher
from helpers.price import adjust_precision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.FileHandler("flux_trading.log"),
    #     logging.StreamHandler()
    # ]
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram notifications for orderblock trading"""

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if enabled and bot_token and chat_id:
            self.send_message(
                "🚀 <b>Enhanced OrderBlock Trading Bot Started</b>")

    def send_message(self, message: str, topic_id: str = None):
        if not self.enabled or not self.bot_token or not self.chat_id:
            return

        try:
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            if topic_id:
                data["message_thread_id"] = topic_id
            response = requests.post(self.base_url, data=data)
            if response.status_code != 200:
                logger.error(f"Telegram error: {response.text}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def notify_orderblock_signal(self, symbol: str, ob: OrderBlockInfo, entry_price: float,
                                 stop_loss: float, take_profit: float, position_size: float,
                                 margin_required: float = None, risk_amount: float = None,
                                 leverage: int = 10, algorithm: str = "flux"):
        """Notify about new order block signal"""
        emoji = "🟢" if ob.ob_type == "Bull" else "🔴"
        side = "LONG" if ob.ob_type == "Bull" else "SHORT"

        # Calculate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        message = (
            f"{emoji} <b>{algorithm.upper()} ORDER BLOCK {side}</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Entry: <b>${entry_price:.4f}</b>\n"
            f"Stop Loss: <b>${stop_loss:.4f}</b>\n"
            f"Take Profit: <b>${take_profit:.4f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Position Details</b>\n"
            f"Position Size: <b>${position_size:.0f}</b>\n"
        )

        if margin_required:
            message += f"Margin Required: <b>${margin_required:.0f}</b>\n"
            message += f"Leverage: <b>{leverage}x</b>\n"

        if risk_amount:
            message += f"Risk Amount: <b>${risk_amount:.0f}</b>\n"

        message += f"Risk/Reward: <b>{rr_ratio:.2f}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"

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

        self.send_message(message, topic_id=os.getenv(
            'TELEGRAM_ORDERS_TOPIC_ID', "5"))

    def notify_order_cancelled(self, symbol: str, reason: str, order_info: str):
        """Notify about order cancellation"""
        message = (
            f"❌ <b>Order Cancelled</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Reason: <b>{reason}</b>\n"
            f"Details: {order_info}"
        )
        self.send_message(message, topic_id=os.getenv(
            'TELEGRAM_SIGNALS_TOPIC_ID', "6"))

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
        self.send_message(message, topic_id=os.getenv(
            'TELEGRAM_SIGNALS_TOPIC_ID', "6"))

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
        self.send_message(message, topic_id=os.getenv(
            'TELEGRAM_SIGNALS_TOPIC_ID', "6"))


def enhanced_analyze_trend_confluence(df: pd.DataFrame, current_idx: int,
                                      higher_tf_data: pd.DataFrame = None):
    """
    Enhanced trend analysis with 4h timeframe support

    Parameters:
    - df: Primary timeframe data
    - current_idx: Current bar index
    - higher_tf_data: 4h timeframe data for confluence

    Returns:
    - dict: Enhanced trend analysis results
    """
    trend_analysis = {
        'primary_trend': 'neutral',
        'trend_strength': 0,
        'confluence_score': 0,
        'trend_change_detected': False,
        'supporting_timeframes': 0,
        'htf_trend': 'neutral',
        'htf_strength': 0
    }

    if current_idx < 50:
        return trend_analysis

    # Calculate EMAs for trend analysis on primary timeframe
    ema_periods = [21, 50, 100]
    trend_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}

    for period in ema_periods:
        if current_idx >= period:
            ema = df['close'].rolling(window=period).mean()
            current_price = df['close'].iloc[current_idx]
            ema_current = ema.iloc[current_idx]
            ema_prev = ema.iloc[current_idx -
                                5] if current_idx >= 5 else ema_current

            # Trend direction based on price vs EMA and EMA slope
            if current_price > ema_current and ema_current > ema_prev:
                trend_votes['bullish'] += 1
            elif current_price < ema_current and ema_current < ema_prev:
                trend_votes['bearish'] += 1
            else:
                trend_votes['neutral'] += 1

    # Determine primary trend
    max_votes = max(trend_votes.values())
    if trend_votes['bullish'] == max_votes and trend_votes['bullish'] >= 2:
        trend_analysis['primary_trend'] = 'bullish'
        trend_analysis['trend_strength'] = (trend_votes['bullish'] / 3) * 100
    elif trend_votes['bearish'] == max_votes and trend_votes['bearish'] >= 2:
        trend_analysis['primary_trend'] = 'bearish'
        trend_analysis['trend_strength'] = (trend_votes['bearish'] / 3) * 100

    # Analyze higher timeframe (4h) if provided
    if higher_tf_data is not None and len(higher_tf_data) > 0:
        htf_ema_21 = higher_tf_data['close'].rolling(window=21).mean()
        htf_ema_50 = higher_tf_data['close'].rolling(window=50).mean()

        if len(htf_ema_21) > 1 and len(htf_ema_50) > 1:
            htf_current_price = higher_tf_data['close'].iloc[-1]
            htf_ema21_current = htf_ema_21.iloc[-1]
            htf_ema50_current = htf_ema_50.iloc[-1]

            # HTF trend determination
            if htf_current_price > htf_ema21_current > htf_ema50_current:
                trend_analysis['htf_trend'] = 'bullish'
                trend_analysis['htf_strength'] = 80
            elif htf_current_price < htf_ema21_current < htf_ema50_current:
                trend_analysis['htf_trend'] = 'bearish'
                trend_analysis['htf_strength'] = 80
            elif htf_current_price > htf_ema21_current or htf_current_price > htf_ema50_current:
                trend_analysis['htf_trend'] = 'bullish'
                trend_analysis['htf_strength'] = 50
            elif htf_current_price < htf_ema21_current or htf_current_price < htf_ema50_current:
                trend_analysis['htf_trend'] = 'bearish'
                trend_analysis['htf_strength'] = 50

            # Check confluence between timeframes
            supporting_count = 0
            if trend_analysis['primary_trend'] == trend_analysis['htf_trend']:
                supporting_count = 3  # Strong confluence
            elif trend_analysis['primary_trend'] != 'neutral' and trend_analysis['htf_trend'] != 'neutral':
                supporting_count = 0  # Conflicting trends
            else:
                supporting_count = 1  # Neutral confluence

            trend_analysis['supporting_timeframes'] = supporting_count

    # Calculate overall confluence score
    if trend_analysis['htf_trend'] != 'neutral':
        if trend_analysis['primary_trend'] == trend_analysis['htf_trend']:
            trend_analysis['confluence_score'] = 85 + \
                (trend_analysis['htf_strength'] * 0.15)
        else:
            trend_analysis['confluence_score'] = 25  # Conflicting trends
    else:
        trend_analysis['confluence_score'] = 50  # Neutral HTF

    # Enhanced trend change detection
    if current_idx >= 10:
        recent_highs = df['high'].iloc[current_idx-10:current_idx+1].max()
        recent_lows = df['low'].iloc[current_idx-10:current_idx+1].min()
        current_close = df['close'].iloc[current_idx]

        # Check for potential reversal patterns
        if (trend_analysis['primary_trend'] == 'bearish' and
                current_close > (recent_lows + (recent_highs - recent_lows) * 0.7)):
            trend_analysis['trend_change_detected'] = True
        elif (trend_analysis['primary_trend'] == 'bullish' and
              current_close < (recent_lows + (recent_highs - recent_lows) * 0.3)):
            trend_analysis['trend_change_detected'] = True

    return trend_analysis


class EnhancedOrderBlockStrategy:
    """Enhanced strategy class for Order Block trading with algorithm choice"""

    def __init__(self, symbol: str, config: Dict):
        self.symbol = symbol
        self.config = config
        self.last_orderblocks = []
        self.last_analysis_time = None
        self.algorithm = config.get(
            'orderblock_algorithm', 'flux')  # 'flux' or 'breaker'

        logger.info(
            f"{symbol}: Using {self.algorithm.upper()} orderblock algorithm")

    def analyze_and_generate_orders(self, data: pd.DataFrame, htf_data: pd.DataFrame = None) -> List[Dict]:
        """
        Analyze market using chosen Order Block algorithm and generate trading orders

        Parameters:
        - data: Primary timeframe OHLCV DataFrame
        - htf_data: Higher timeframe (4h) data for confluence

        Returns:
        - List of order dictionaries
        """
        try:
            # Choose algorithm based on config
            if self.algorithm == 'flux':
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
            elif self.algorithm == 'breaker':
                order_blocks = detect_order_breaker_blocks(
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
            else:
                logger.error(f"Unknown orderblock algorithm: {self.algorithm}")
                return []

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
                order = self._create_enhanced_order_from_orderblock(
                    ob, current_price, data, htf_data)
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

    def _create_enhanced_order_from_orderblock(self, ob: OrderBlockInfo, current_price: float,
                                               data: pd.DataFrame, htf_data: pd.DataFrame = None) -> Optional[Dict]:
        """Create enhanced trading order from order block with better risk management"""
        try:
            # Calculate ATR for volatility-based risk management
            atr_period = 14
            if len(data) >= atr_period:
                data['tr'] = np.maximum(
                    data['high'] - data['low'],
                    np.maximum(
                        abs(data['high'] - data['close'].shift(1)),
                        abs(data['low'] - data['close'].shift(1))
                    )
                )
                current_atr = data['tr'].rolling(atr_period).mean().iloc[-1]
            else:
                current_atr = abs(data['high'].iloc[-1] - data['low'].iloc[-1])

            # Enhanced entry price calculation - consider OB structure
            ob_height = abs(ob.top - ob.bottom)

            if ob.ob_type == "Bull":
                side = "LONG"
                # Enter slightly above bottom for better fill probability
                entry_buffer = min(ob_height * 0.1, current_atr * 0.1)
                entry_price = ob.bottom + entry_buffer

                # Enhanced stop loss calculation
                # Consider: OB structure, ATR, and minimum risk/reward
                atr_multiplier = self.config.get('atr_stop_multiplier', 1.5)
                min_stop_distance = current_atr * atr_multiplier
                ob_based_stop = ob.bottom - (ob_height * 0.2)  # 20% below OB

                # Use the more conservative (further) stop loss
                stop_loss = min(ob_based_stop, entry_price - min_stop_distance)

                # Ensure minimum risk/reward ratio
                min_rr = self.config.get('min_risk_reward', 2.0)
                risk_distance = entry_price - stop_loss
                take_profit = entry_price + (risk_distance * min_rr)

            else:  # Bear
                side = "SHORT"
                # Enter slightly below top for better fill probability
                entry_buffer = min(ob_height * 0.1, current_atr * 0.1)
                entry_price = ob.top - entry_buffer

                # Enhanced stop loss calculation
                atr_multiplier = self.config.get('atr_stop_multiplier', 1.5)
                min_stop_distance = current_atr * atr_multiplier
                ob_based_stop = ob.top + (ob_height * 0.2)  # 20% above OB

                # Use the more conservative (further) stop loss
                stop_loss = max(ob_based_stop, entry_price + min_stop_distance)

                # Ensure minimum risk/reward ratio
                min_rr = self.config.get('min_risk_reward', 2.0)
                risk_distance = stop_loss - entry_price
                take_profit = entry_price - (risk_distance * min_rr)

            # Check distance from current price
            distance_pct = abs((entry_price / current_price) - 1) * 100
            max_distance = self.config.get('max_distance_pct', 1.5)

            if distance_pct > max_distance:
                logger.info(
                    f"{self.symbol}: OB too far ({distance_pct:.1f}% > {max_distance}%) - skipping")
                return None

            # Enhanced position sizing with volatility consideration
            capital_per_symbol = self.config.get('capital_per_symbol', 500)
            leverage = self.config.get('leverage', 10)

            # Base position calculation
            capital_usage_pct = self.config.get(
                'capital_usage_pct', 15.0) / 100
            base_position_value = capital_per_symbol * capital_usage_pct

            # Volatility adjustment - reduce size in high volatility
            avg_atr = data['tr'].rolling(20).mean(
            ).iloc[-1] if len(data) >= 20 else current_atr
            volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

            volatility_adjustment = 1.0
            if volatility_ratio > 1.5:  # High volatility
                volatility_adjustment = 0.7
            elif volatility_ratio > 1.2:  # Moderate volatility
                volatility_adjustment = 0.85

            # Apply leverage and volatility adjustment
            leveraged_position_size = base_position_value * leverage * volatility_adjustment

            # Risk management: ensure acceptable risk per trade
            price_risk_pct = abs((entry_price - stop_loss) / entry_price)
            max_risk_pct = self.config.get('max_risk_per_trade_pct', 6.0) / 100

            if price_risk_pct > max_risk_pct:
                risk_adjustment = max_risk_pct / price_risk_pct
                leveraged_position_size *= risk_adjustment
                logger.info(
                    f"{self.symbol}: High risk trade ({price_risk_pct*100:.1f}%), reducing position by {(1-risk_adjustment)*100:.1f}%")

            # Quality-based position sizing
            if hasattr(ob, 'entry_score'):
                quality_multiplier = 1.0
                if ob.entry_score >= 85:
                    quality_multiplier = 1.4  # +40% for excellent setups
                elif ob.entry_score >= 70:
                    quality_multiplier = 1.2  # +20% for good setups
                elif ob.entry_score < 50:
                    quality_multiplier = 0.6   # -40% for moderate setups

                leveraged_position_size *= quality_multiplier

            # Trend confluence adjustment
            trend_confluence_bonus = 1.0
            if hasattr(ob, 'trend_confluence') and htf_data is not None:
                conf = ob.trend_confluence
                if conf.get('supporting_timeframes', 0) >= 2:
                    trend_confluence_bonus = 1.15  # +15% for good confluence

            leveraged_position_size *= trend_confluence_bonus

            position_size = leveraged_position_size
            margin_required = position_size / leverage
            risk_amount = position_size * price_risk_pct

            logger.info(
                f"{self.symbol}: Enhanced position sizing: ${position_size:.0f} "
                f"(Volatility adj: {volatility_adjustment:.2f}, Risk: {price_risk_pct*100:.1f}%)")

            order = {
                'symbol': self.symbol,
                'side': side,
                'type': 'LIMIT',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'margin_required': margin_required,
                'risk_amount': risk_amount,
                'leverage': leverage,
                'order_block': ob,
                'created_time': datetime.now(),
                'distance_pct': distance_pct,
                'atr_at_creation': current_atr,
                'volatility_ratio': volatility_ratio,
                'algorithm': self.algorithm
            }

            return order

        except Exception as e:
            logger.error(f"Error creating enhanced order from OB: {e}")
            return None

    def should_cancel_order(self, order: Dict, current_data: pd.DataFrame,
                            htf_data: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Determine if an order should be cancelled based on market conditions

        Returns:
        - (should_cancel: bool, reason: str)
        """
        try:
            ob = order['order_block']
            current_price = current_data['close'].iloc[-1]
            current_idx = len(current_data) - 1

            # Check if order block has been broken
            if hasattr(ob, 'breaker') and ob.breaker:
                return True, "Order block broken"

            # Check if order is too old
            order_age = (datetime.now() -
                         order['created_time']).total_seconds() / 3600
            max_age = self.config.get('max_order_age_hours', 24)
            if order_age > max_age:
                return True, f"Order expired ({order_age:.1f}h > {max_age}h)"

            # Check if price has moved too far from entry
            distance_pct = abs(
                (order['entry_price'] / current_price) - 1) * 100
            max_distance = self.config.get('max_distance_cancel_pct', 3.0)
            if distance_pct > max_distance:
                return True, f"Price too far from entry ({distance_pct:.1f}% > {max_distance}%)"

            # Check trend change using enhanced analysis
            if htf_data is not None:
                trend_analysis = enhanced_analyze_trend_confluence(
                    current_data, current_idx, htf_data)

                # Cancel if trend has changed significantly
                if hasattr(ob, 'trend_confluence'):
                    original_trend = ob.trend_confluence.get(
                        'primary_trend', 'neutral')
                    current_trend = trend_analysis['primary_trend']

                    # Cancel if trends are now opposing
                    if ((original_trend == 'bullish' and current_trend == 'bearish') or
                            (original_trend == 'bearish' and current_trend == 'bullish')):
                        return True, f"Trend changed from {original_trend} to {current_trend}"

            # Check volatility increase
            if 'atr_at_creation' in order:
                current_atr = current_data['high'].iloc[-1] - \
                    current_data['low'].iloc[-1]
                atr_increase = current_atr / order['atr_at_creation']
                if atr_increase > 2.0:  # ATR doubled
                    return True, f"Volatility increased significantly ({atr_increase:.1f}x)"

            # Check if stop loss would now be too tight
            if order['side'] == 'LONG':
                new_stop_suggested = current_price * \
                    (1 - self.config.get('emergency_stop_pct', 0.03))
                if new_stop_suggested > order['stop_loss']:
                    return True, "Stop loss would be too tight for current volatility"
            else:
                new_stop_suggested = current_price * \
                    (1 + self.config.get('emergency_stop_pct', 0.03))
                if new_stop_suggested < order['stop_loss']:
                    return True, "Stop loss would be too tight for current volatility"

            return False, ""

        except Exception as e:
            logger.error(
                f"Error checking order cancellation for {order.get('symbol', 'unknown')}: {e}")
            return False, ""


class EnhancedOrderBlockBot:
    """Enhanced live trading bot using Order Block strategy with algorithm choice"""

    def __init__(self, symbols: List[str], config: Dict):
        """
        Initialize Enhanced Order Block trading bot

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
        self.htf_data_cache = {}  # Cache for 4h data

        # Initialize strategies
        for symbol in symbols:
            self.strategies[symbol] = EnhancedOrderBlockStrategy(
                symbol, config)
            self.active_orders[symbol] = []
            self.open_positions[symbol] = []

        # Get initial capital
        self.initial_capital = self._get_usdt_balance()
        capital_per_symbol = self.initial_capital / len(symbols)
        self.config['capital_per_symbol'] = capital_per_symbol

        algorithm = config.get('orderblock_algorithm', 'flux').upper()
        logger.info(
            f"Initialized Enhanced OrderBlock Bot ({algorithm}) for {len(symbols)} symbols")
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

    def _fetch_htf_data(self, symbol: str) -> pd.DataFrame:
        """Fetch 4h timeframe data for trend confluence"""
        try:
            # Cache 4h data to avoid excessive API calls
            cache_key = f"{symbol}_4h"
            current_time = datetime.now()

            # Refresh cache every 30 minutes
            if (cache_key in self.htf_data_cache and
                    (current_time - self.htf_data_cache[cache_key]['timestamp']).total_seconds() < 1800):
                return self.htf_data_cache[cache_key]['data']

            start_time = datetime.now() - timedelta(days=30)  # 30 days of 4h data

            htf_data = self.data_fetcher.get_historical_klines(
                symbol=symbol,
                interval='4h',
                start_time=start_time
            )

            # Cache the data
            self.htf_data_cache[cache_key] = {
                'data': htf_data,
                'timestamp': current_time
            }

            return htf_data
        except Exception as e:
            logger.error(f"Error fetching 4h data for {symbol}: {e}")
            return pd.DataFrame()

    def _analyze_symbol(self, symbol: str) -> List[Dict]:
        """Analyze single symbol using Enhanced Order Block strategy"""
        try:
            # Fetch latest data
            data = self._fetch_data(symbol)
            if data.empty or len(data) < 100:
                logger.warning(f"{symbol}: Insufficient data for analysis")
                return []

            # Fetch 4h data for trend confluence
            htf_data = self._fetch_htf_data(symbol)

            # Run Enhanced Order Block analysis
            orders = self.strategies[symbol].analyze_and_generate_orders(
                data, htf_data)

            # Add symbol and additional info to orders
            for order in orders:
                order['symbol'] = symbol
                order['current_price'] = self.current_prices[symbol]

            return orders

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return []

    def _check_order_cancellations(self):
        """Check if any active orders should be cancelled based on market conditions"""
        for symbol in self.symbols:
            try:
                if not self.active_orders[symbol]:
                    continue

                # Fetch current data for analysis
                current_data = self._fetch_data(symbol)
                htf_data = self._fetch_htf_data(symbol)

                if current_data.empty:
                    continue

                orders_to_cancel = []
                for order in self.active_orders[symbol]:
                    should_cancel, reason = self.strategies[symbol].should_cancel_order(
                        order, current_data, htf_data)

                    if should_cancel:
                        orders_to_cancel.append((order, reason))

                # Cancel the flagged orders
                for order, reason in orders_to_cancel:
                    self._cancel_order(order, reason)

            except Exception as e:
                logger.error(
                    f"Error checking order cancellations for {symbol}: {e}")

    def _cancel_order(self, order: Dict, reason: str):
        """Cancel an active order"""
        try:
            symbol = order['symbol']

            logger.info(
                f"🚫 CANCELLING ORDER: {symbol} {order['side']} - {reason}")

            # Cancel on exchange if not in test mode
            if not self.config.get('test_mode', True) and 'order_id' in order:
                try:
                    cancel_response = self.client.futures_cancel_order(
                        symbol=symbol,
                        orderId=order['order_id']
                    )
                    logger.info(
                        f"   ✅ Order cancelled on exchange: {cancel_response.get('orderId', 'N/A')}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to cancel on exchange: {e}")

            # Remove from active orders
            if order in self.active_orders[symbol]:
                self.active_orders[symbol].remove(order)

            # Send notification
            order_info = f"{order['side']} @ ${order['entry_price']:.4f}"
            self.telegram.notify_order_cancelled(symbol, reason, order_info)

            logger.info(
                f"✅ Order removed from tracking: {symbol} {order['side']}")

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")

    def _place_order(self, order: Dict) -> bool:
        """Place order on exchange"""
        try:
            symbol = order['symbol']
            ob = order['order_block']
            algorithm = order.get('algorithm', 'flux')

            logger.info(
                f"🎯 PLACING {algorithm.upper()} ORDER: {symbol} {order['side']} at ${order['entry_price']:.4f}")
            logger.info(
                f"   Distance from current: {order['distance_pct']:.2f}%")
            logger.info(f"   Position size: ${order['position_size']:.0f}")
            logger.info(f"   Stop loss: ${order['stop_loss']:.4f}")
            logger.info(f"   Take profit: ${order['take_profit']:.4f}")
            logger.info(
                f"   Risk/Reward: {abs(order['take_profit'] - order['entry_price']) / abs(order['entry_price'] - order['stop_loss']):.2f}")

            # Send Telegram notification
            self.telegram.notify_orderblock_signal(
                symbol=symbol,
                ob=ob,
                entry_price=order['entry_price'],
                stop_loss=order['stop_loss'],
                take_profit=order['take_profit'],
                position_size=order['position_size'],
                margin_required=order['margin_required'],
                risk_amount=order['risk_amount'],
                leverage=order['leverage'],
                algorithm=algorithm
            )

            if self.config.get('test_mode', True):
                logger.info(
                    f"✅ TEST MODE: {symbol} {order['side']} limit order created")
                logger.info(f"   Entry: ${order['entry_price']:.4f}")
                logger.info(
                    f"   Stop Loss: ${order['stop_loss']:.4f} (-{((order['entry_price']-order['stop_loss'])/order['entry_price']*100):.1f}%)")
                logger.info(
                    f"   Take Profit: ${order['take_profit']:.4f} (+{((order['take_profit']-order['entry_price'])/order['entry_price']*100):.1f}%)")
                order['status'] = 'TEST'
                order['order_id'] = f"TEST_{int(time.time())}"
                self.active_orders[symbol].append(order)
                return True

            # LIVE MODE - Place actual orders
            logger.info(
                f"🔴 LIVE MODE: Placing {symbol} {order['side']} order on Binance...")

            # Set leverage first
            try:
                leverage_response = self.client.futures_change_leverage(
                    symbol=symbol,
                    leverage=order['leverage']
                )
                logger.info(f"   ✅ Leverage set to {order['leverage']}x")
            except Exception as e:
                logger.warning(f"   ⚠️ Leverage setting failed: {e}")

            # Calculate quantity
            quantity = order['position_size'] / order['entry_price']
            quantity = adjust_precision(
                quantity, self.symbol_info[symbol]['quantityPrecision'])

            if quantity <= 0:
                logger.warning(f"❌ Invalid quantity for {symbol}: {quantity}")
                return False

            # Place main limit order
            logger.info(
                f"   📝 Creating limit order: {quantity:.6f} {symbol} at ${order['entry_price']:.4f}")

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

            logger.info(f"✅ LIMIT ORDER PLACED:")
            logger.info(f"   Order ID: {response['orderId']}")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {order['side']}")
            logger.info(f"   Quantity: {quantity:.6f}")
            logger.info(f"   Price: ${order['entry_price']:.4f}")
            logger.info(
                f"   Stop Loss will be placed after fill: ${order['stop_loss']:.4f}")
            logger.info(
                f"   Take Profit will be placed after fill: ${order['take_profit']:.4f}")

            return True

        except Exception as e:
            logger.error(
                f"❌ ERROR placing order for {order.get('symbol', 'unknown')}: {e}")
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

            logger.info(f"📋 PLACING EXIT ORDERS for {symbol} {order['side']}:")
            logger.info(f"   Quantity: {quantity:.6f}")

            # Place stop loss
            logger.info(
                f"   📉 Creating STOP LOSS at ${order['stop_loss']:.4f}")
            sl_response = self.client.futures_create_order(
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
            logger.info(
                f"   ✅ STOP LOSS placed: Order ID {sl_response['orderId']}")

            # Place take profit
            logger.info(
                f"   📈 Creating TAKE PROFIT at ${order['take_profit']:.4f}")
            tp_response = self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if order['side'] == 'LONG' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=round_step_size(
                    order['take_profit'],
                    float(self.symbol_info[symbol]['tickSize'])
                )
            )
            logger.info(
                f"   ✅ TAKE PROFIT placed: Order ID {tp_response['orderId']}")

            # Store exit order IDs
            order['stop_loss_id'] = sl_response['orderId']
            order['take_profit_id'] = tp_response['orderId']

            logger.info(f"✅ ALL EXIT ORDERS PLACED for {symbol}")

        except Exception as e:
            logger.error(f"❌ ERROR placing exit orders for {symbol}: {e}")

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
        """Enhanced trading loop with order cancellation checks"""
        logger.info("🔄 Starting enhanced trading loop...")

        while self.running:
            try:
                start_time = time.time()

                # Analyze symbols and place new orders
                for symbol in self.symbols:
                    try:
                        if len(self.active_orders[symbol]) >= self.config.get('max_orders_per_symbol', 2):
                            continue

                        orders = self._analyze_symbol(symbol)
                        for order in orders:
                            if self._should_place_order(order):
                                success = self._place_order(order)
                                if success:
                                    time.sleep(1)  # Small delay between orders

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # Check order cancellations (enhanced feature)
                self._check_order_cancellations()

                # Check existing orders and positions
                self._check_orders()
                self._check_positions()

                # Clean old orders
                self._clean_old_orders()

                # Log status
                total_orders = sum(len(orders)
                                   for orders in self.active_orders.values())
                total_positions = sum(len(positions)
                                      for positions in self.open_positions.values())
                logger.info(
                    f"📊 Status: {total_orders} active orders, {total_positions} open positions")

                # Sleep until next cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.get(
                    'scan_interval', 60) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)

    def _should_place_order(self, order: Dict) -> bool:
        """Enhanced order validation"""
        try:
            symbol = order['symbol']
            current_price = self.current_prices.get(symbol, 0)

            # Basic validations
            if current_price == 0:
                return False

            # Check minimum order size
            min_notional = self.config.get('min_order_value', 10)
            if order['position_size'] < min_notional:
                logger.info(
                    f"{symbol}: Order too small (${order['position_size']:.0f} < ${min_notional})")
                return False

            # Check if we already have similar orders
            for existing_order in self.active_orders[symbol]:
                if (existing_order['side'] == order['side'] and
                        abs(existing_order['entry_price'] - order['entry_price']) / order['entry_price'] < 0.005):  # 0.5%
                    logger.info(f"{symbol}: Similar order already exists")
                    return False

            # Enhanced validation: Check if OB is still valid
            ob = order['order_block']
            if hasattr(ob, 'breaker') and ob.breaker:
                logger.info(f"{symbol}: Order block has been broken")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False

    def _clean_old_orders(self):
        """Remove old orders that are no longer relevant"""
        max_age_hours = self.config.get('max_order_age_hours', 24)
        current_time = datetime.now()

        for symbol in self.symbols:
            orders_to_remove = []
            for order in self.active_orders[symbol]:
                age_hours = (current_time -
                             order['created_time']).total_seconds() / 3600
                if age_hours > max_age_hours:
                    orders_to_remove.append(order)

            for order in orders_to_remove:
                self._cancel_order(order, f"Order expired ({age_hours:.1f}h)")

    def start(self):
        """Start the enhanced trading bot"""
        if self.running:
            logger.warning("Bot is already running")
            return

        logger.info("🚀 Starting Enhanced OrderBlock Trading Bot...")

        # Log configuration
        algorithm = self.config.get('orderblock_algorithm', 'flux').upper()
        test_mode = "TEST MODE" if self.config.get(
            'test_mode', True) else "LIVE MODE"
        logger.info(f"🔧 Configuration: {algorithm} algorithm, {test_mode}")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(
            f"💰 Capital per symbol: ${self.config['capital_per_symbol']:.0f}")
        logger.info(f"🎯 Leverage: {self.config.get('leverage', 10)}x")
        logger.info(
            f"⏰ Scan interval: {self.config.get('scan_interval', 60)}s")

        self.running = True
        trading_thread = threading.Thread(
            target=self._trading_loop, daemon=True)
        trading_thread.start()

        logger.info("✅ Enhanced OrderBlock Trading Bot started successfully!")

    def stop(self):
        """Stop the trading bot"""
        if not self.running:
            logger.warning("Bot is not running")
            return

        logger.info("🛑 Stopping Enhanced OrderBlock Trading Bot...")
        self.running = False
        time.sleep(2)
        logger.info("✅ Bot stopped successfully")

    def get_status(self) -> Dict:
        """Get current bot status"""
        total_orders = sum(len(orders)
                           for orders in self.active_orders.values())
        total_positions = sum(len(positions)
                              for positions in self.open_positions.values())

        status = {
            'running': self.running,
            'algorithm': self.config.get('orderblock_algorithm', 'flux'),
            'symbols': self.symbols,
            'total_capital': self.initial_capital,
            'capital_per_symbol': self.config['capital_per_symbol'],
            'active_orders': total_orders,
            'open_positions': total_positions,
            'test_mode': self.config.get('test_mode', True),
            'orders_by_symbol': {symbol: len(orders) for symbol, orders in self.active_orders.items()},
            'positions_by_symbol': {symbol: len(positions) for symbol, positions in self.open_positions.items()},
            'current_prices': self.current_prices.copy()
        }

        return status


def load_config() -> Dict:
    """Load enhanced trading configuration"""
    config = {
        # Algorithm choice
        'orderblock_algorithm': 'flux',  # 'flux' or 'breaker'

        # Trading parameters
        'test_mode': True,
        'test_capital': 1000.0,
        'max_capital': 5000.0,
        'interval': '15m',
        'lookback_hours': 168,
        'scan_interval': 60,

        # OrderBlock detection
        'swing_length': 10,
        'max_atr_mult': 3.5,
        'mitigation_method': 'Wick',
        'max_bullish_obs': 5,
        'max_bearish_obs': 5,
        'use_entry_evaluation': True,
        'entry_threshold': 45,

        # Enhanced risk management
        'leverage': 10,
        'capital_usage_pct': 15.0,
        'max_risk_per_trade_pct': 6.0,
        'atr_stop_multiplier': 1.5,
        'min_risk_reward': 2.0,
        'emergency_stop_pct': 3.0,

        # Order management
        'max_orders_per_symbol': 2,
        'max_distance_pct': 1.5,
        'max_distance_cancel_pct': 3.0,
        'max_order_age_hours': 24,
        'min_order_value': 10,

        # Notifications
        'telegram_enabled': True
    }

    # Override with environment variables if available
    if os.getenv('ORDERBLOCK_ALGORITHM'):
        config['orderblock_algorithm'] = os.getenv(
            'ORDERBLOCK_ALGORITHM').lower()
    if os.getenv('TRADING_MODE'):
        config['test_mode'] = os.getenv('TRADING_MODE').lower() != 'live'
    if os.getenv('LEVERAGE'):
        config['leverage'] = int(os.getenv('LEVERAGE'))
    if os.getenv('CAPITAL_USAGE_PCT'):
        config['capital_usage_pct'] = float(os.getenv('CAPITAL_USAGE_PCT'))

    return config


def main():
    """Enhanced main function with algorithm choice"""
    load_dotenv()

    import argparse
    parser = argparse.ArgumentParser(
        description='Enhanced OrderBlock Live Trading Bot')
    parser.add_argument('--symbols', type=str, default='BTCUSDT,ETHUSDT,SOLUSDT',
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--algorithm', type=str, choices=['flux', 'breaker'], default='flux',
                        help='OrderBlock algorithm to use (flux or breaker)')
    parser.add_argument('--mode', type=str, choices=['test', 'live'], default='test',
                        help='Trading mode (test or live)')
    parser.add_argument('--leverage', type=int, default=10,
                        help='Leverage to use')
    parser.add_argument('--capital-usage', type=float, default=15.0,
                        help='Capital usage percentage per trade')
    parser.add_argument('--scan-interval', type=int, default=60,
                        help='Scan interval in seconds')

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Load and update config
    config = load_config()
    config['orderblock_algorithm'] = args.algorithm
    config['test_mode'] = args.mode == 'test'
    config['leverage'] = args.leverage
    config['capital_usage_pct'] = args.capital_usage
    config['scan_interval'] = args.scan_interval

    logger.info("🚀 ENHANCED ORDERBLOCK TRADING BOT")
    logger.info("=" * 50)
    logger.info(f"Algorithm: {args.algorithm.upper()}")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Leverage: {args.leverage}x")
    logger.info(f"Capital Usage: {args.capital_usage}% per trade")

    # Create and start bot
    bot = EnhancedOrderBlockBot(symbols=symbols, config=config)

    try:
        bot.start()

        # Keep the main thread alive
        while bot.running:
            time.sleep(10)

            # Print status every 5 minutes
            if int(time.time()) % 300 == 0:
                status = bot.get_status()
                logger.info(
                    f"📊 Bot Status: {status['active_orders']} orders, {status['open_positions']} positions")

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
