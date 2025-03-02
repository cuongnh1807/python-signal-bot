import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from strategy import analyze_trading_setup, calculate_rsi, calculate_macd
from smartmoneyconcepts.smc import smc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FuturesStrategy:
    """
    Futures trading strategy based on SMC (Smart Money Concepts) analysis.
    Includes order generation and backtesting capabilities.
    """

    def __init__(self,
                 initial_capital: float = 1000.0,
                 max_risk_per_trade: float = 0.02,
                 take_profit_levels: Dict[str, float] = None,
                 default_leverage: int = 10):
        """
        Initialize the futures trading strategy.

        Parameters:
        -----------
        initial_capital: Starting capital for backtesting
        max_risk_per_trade: Maximum risk per trade as a percentage of capital (0.02 = 2%)
        take_profit_levels: Dictionary with TP levels (e.g., {'tp1': 0.5, 'tp2': 1.0, 'tp3': 1.5})
        default_leverage: Default leverage to use when not specified by the strategy
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.default_leverage = default_leverage

        # Default take profit levels if none provided
        self.take_profit_levels = take_profit_levels or {
            'tp1': 0.5,  # 50% of the distance from entry to stop loss
            'tp2': 1.0,  # 100% of the distance from entry to stop loss
            'tp3': 1.5   # 150% of the distance from entry to stop loss
        }

        # Track open positions and trade history
        self.open_positions = []
        self.trade_history = []
        self.current_trades = {}

    def analyze_market(self, data: pd.DataFrame, ignore_old_ob: bool = True) -> Dict:
        """
        Analyze market data and generate trading setups.

        Parameters:
        -----------
        data: DataFrame with OHLCV data

        Returns:
        --------
        Dict containing trading setups and analysis
        """
        # Calculate swing highs/lows
        swing_hl = smc.swing_highs_lows(data, swing_length=5)

        # Get trading setups
        return analyze_trading_setup(data, swing_hl, ignore_old_ob)

    def generate_orders(self, analysis: Dict, min_setup_quality: float = 65.0,
                        min_volume_ratio: float = 2.0,
                        respect_pressure: bool = True,
                        respect_warnings: bool = True) -> List[Dict]:
        """
        Generate trading orders with improved entry and stop loss based on volume ratio.
        """
        orders = []
        if 'trade_setups' not in analysis or not analysis['trade_setups']:
            logger.info("No valid trade setups found")
            return orders

        current_price = analysis['current_price']
        market_pressure = analysis.get('volume_analysis', {}).get(
            'analysis', {}).get('pressure', 'Neutral')

        # Calculate volatility for adjustments
        volatility = self._calculate_volatility(analysis.get('price_data'))

        for setup in analysis['trade_setups']:
            # Filter setups by quality and volume
            if setup['setup_quality'] < min_setup_quality:
                continue

            volume_ratio = setup.get('ob_volume_ratio', 0)
            if volume_ratio < min_volume_ratio:
                continue
            # Check for critical warning messages
            if respect_warnings and self._has_critical_warnings(setup):
                logger.info(
                    f"Rejecting setup due to critical warnings: {setup.get('warning_messages', [])}")
                continue

            # Check market pressure alignment
            if respect_pressure and self._is_pressure_contradictory(setup['position_type'], market_pressure):
                logger.info(
                    f"Rejecting setup due to contradictory market pressure: {market_pressure}")
                continue

            # Determine position side
            side = "LONG" if setup['position_type'] == 'LONG' else "SHORT"

            # Extract order block levels
            ob_levels = setup.get('ob_level', '0-0')
            try:
                ob_bottom, ob_top = map(float, ob_levels.split('-'))
            except (ValueError, AttributeError):
                logger.warning(
                    f"Invalid order block level format: {ob_levels}")
                continue

            # Calculate entry prices based on order block levels and volume ratio
            entry_prices = self._calculate_entries_from_ob(
                side, ob_bottom, ob_top, current_price, setup['setup_quality'],
                volume_ratio, volatility)

            # Use the selected entry price
            entry_price = entry_prices['selected']

            # Determine entry type (MARKET or LIMIT)
            if side == "LONG":
                entry_type = "MARKET" if current_price <= entry_price else "LIMIT"
            else:  # SHORT
                entry_type = "MARKET" if current_price >= entry_price else "LIMIT"

            # Calculate stop loss based on order block, volume ratio and volatility
            stop_loss = self._calculate_stop_from_ob(
                side, ob_bottom, ob_top, entry_price, volume_ratio, volatility)

            # Calculate risk distance in price units
            risk_distance = abs(entry_price - stop_loss)

            # Fixed leverage at 20x
            leverage = 20
            # Calculate base margin percentage based on setup quality and volume ratio
            # Higher quality setups get higher percentage of capital
            base_percent = 0.1  # Start with 10% of capital
            # Adjust based on setup quality (0-20% bonus)
            quality_bonus = (setup['setup_quality'] -
                             65) / 100  # 65 is min quality

            # Adjust based on volume ratio (0-15% bonus)
            volume_bonus = min(volume_ratio / 20, 0.15)

            # Calculate total margin percentage (10-45% of capital)
            margin_percent = min(
                base_percent + quality_bonus + volume_bonus, 0.45)

            # Calculate margin amount based on current capital
            margin_amount = self.capital * margin_percent

            # Ensure margin is between $100 and $500
            margin_amount = max(100, min(500, margin_amount))

            # Calculate position size based on margin and leverage
            position_size = margin_amount * leverage

            # Calculate maximum acceptable loss (10% of margin)
            max_acceptable_loss = margin_amount * 0.1

            # Check if risk is acceptable
            price_risk_per_unit = risk_distance
            total_price_risk = price_risk_per_unit * \
                (position_size / entry_price)

            # If risk is too high, adjust position size down
            if total_price_risk > max_acceptable_loss:
                adjustment_factor = max_acceptable_loss / total_price_risk
                position_size = position_size * adjustment_factor
                margin_amount = position_size / leverage
                logger.debug(
                    f"Adjusted position size down due to high risk. New margin: ${margin_amount:.2f}")

            # Create order
            order = {
                'symbol': setup.get('symbol', 'UNKNOWN'),
                'side': side,
                'entry_type': entry_type,
                'entry_price': entry_price,
                'entry_options': entry_prices,
                'stop_loss': stop_loss,
                'take_profit': self._calculate_dynamic_take_profits(
                    entry_price, stop_loss, side, setup['setup_type'], volume_ratio),
                'position_size': position_size,
                'leverage': leverage,
                'setup_quality': setup['setup_quality'],
                'setup_type': setup['setup_type'],
                'ob_levels': {'bottom': ob_bottom, 'top': ob_top},
                'timestamp': datetime.now(),
                'status': 'PENDING',
                'margin_amount': margin_amount,
                'margin_percent': margin_percent * 100,  # Store as percentage for reference
                'max_loss_amount': max_acceptable_loss,
                'volume_ratio': volume_ratio,
                'market_pressure': market_pressure,
                'warning_messages': setup.get('warning_messages', []),
                'risk_reward_ratio': {level: abs(price - entry_price) / risk_distance
                                      for level, price in self._calculate_dynamic_take_profits(
                                          entry_price, stop_loss, side, setup['setup_type'], volume_ratio).items()}
            }

            orders.append(order)

        return orders

    def _calculate_entries_from_ob(self, side: str, ob_bottom: float, ob_top: float,
                                   current_price: float, setup_quality: float,
                                   volume_ratio: float,
                                   volatility: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate entry prices based on order block levels and volume ratio.

        Returns:
        --------
        Dictionary with aggressive, moderate, and conservative entry prices
        """
        # Calculate order block midpoint
        ob_mid = (ob_top + ob_bottom) / 2
        ob_height = ob_top - ob_bottom

        # Adjust entry aggression based on volume ratio and setup quality
        # Higher volume = more confident in the level = more aggressive entry
        if volume_ratio >= 8 and setup_quality >= 85:
            aggression = 0.9  # Very aggressive for high volume & quality
        elif volume_ratio >= 5 or setup_quality >= 80:
            aggression = 0.7  # Aggressive for good volume or quality
        elif volume_ratio >= 3 or setup_quality >= 70:
            aggression = 0.5  # Moderate for decent volume or quality
        else:
            aggression = 0.3  # Conservative for low volume and quality

        entries = {}

        if side == "LONG":
            # For long positions, entries are near the bottom of the order block
            entries['aggressive'] = ob_bottom + (ob_height * 0.1)
            entries['moderate'] = ob_bottom + (ob_height * 0.3)
            entries['conservative'] = ob_mid

            # If current price is already inside the order block
            if current_price > ob_bottom and current_price < ob_top:
                entries['aggressive'] = current_price
                entries['moderate'] = current_price

            # Select entry based on aggression level
            entries['selected'] = ob_bottom + (ob_height * aggression)
        else:
            # For short positions, entries are near the top of the order block
            entries['aggressive'] = ob_top - (ob_height * 0.1)
            entries['moderate'] = ob_top - (ob_height * 0.3)
            entries['conservative'] = ob_mid

            # If current price is already inside the order block
            if current_price > ob_bottom and current_price < ob_top:
                entries['aggressive'] = current_price
                entries['moderate'] = current_price

            # Select entry based on aggression level
            entries['selected'] = ob_top - (ob_height * aggression)

        return entries

    def _calculate_stop_from_ob(self, side: str, ob_bottom: float, ob_top: float,
                                entry_price: float, volume_ratio: float,
                                volatility: Optional[float] = None) -> float:
        """
        Calculate stop loss based on volume ratio and volatility for better protection.

        Parameters:
        -----------
        side: Position side (LONG or SHORT)
        ob_bottom: Bottom price of order block
        ob_top: Top price of order block
        entry_price: Entry price
        volume_ratio: Volume ratio of OB compared to average
        volatility: Market volatility (optional)

        Returns:
        --------
        Stop loss price with better protection against premature stopouts
        """
        ob_height = ob_top - ob_bottom

        # Adjust buffer based on volume ratio - higher volume = wider stop
        # Volume ratio > 10 indicates very significant level that may have more volatility
        if volume_ratio >= 10:
            buffer_ratio = 0.6  # Wider stop for very high volume OB
        elif volume_ratio >= 5:
            buffer_ratio = 0.5  # Wide buffer for good volume OB
        elif volume_ratio >= 3:
            buffer_ratio = 0.4  # Medium buffer for moderate volume OB
        else:
            buffer_ratio = 0.3  # Tighter buffer for low volume OB

        # Use volatility to ensure minimum stop distance
        min_stop_distance = 0
        if volatility is not None:
            # Scale volatility buffer with volume ratio
            # Higher volume = higher multiplier
            vol_multiplier = 2.0 + (volume_ratio * 0.2)
            min_stop_distance = volatility * vol_multiplier

        # Calculate stop distance
        stop_distance = max(ob_height * buffer_ratio, min_stop_distance)

        if side == "LONG":
            # For long positions
            if entry_price <= ob_bottom:
                # Entry below OB
                return entry_price - stop_distance
            elif entry_price <= ob_top:
                # Entry inside OB
                return ob_bottom - (ob_height * buffer_ratio)
            else:
                # Entry above OB
                return ob_bottom - (ob_height * buffer_ratio)
        else:  # SHORT
            # For short positions
            if entry_price >= ob_top:
                # Entry above OB
                return entry_price + stop_distance
            elif entry_price >= ob_bottom:
                # Entry inside OB
                return ob_top + (ob_height * buffer_ratio)
            else:
                # Entry below OB
                return ob_top + (ob_height * buffer_ratio)

    def _calculate_volatility(self, price_data):
        """Calculate recent market volatility"""
        if price_data is None or len(price_data) < 20:
            return None

        # Simple ATR calculation
        high = price_data['high'].values[-20:]
        low = price_data['low'].values[-20:]
        close = price_data['close'].values[-20:]

        ranges = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
        ranges = np.maximum(ranges, np.abs(low[1:] - close[:-1]))
        return np.mean(ranges)

    def _is_pressure_contradictory(self, position_type: str, market_pressure: str) -> bool:
        """
        Check if market pressure contradicts position direction.

        Returns:
        - True if pressure contradicts position, False otherwise
        """
        # Check for strong contradictory pressure
        if position_type == 'LONG' and any(term in market_pressure for term in
                                           ['Strong Selling', 'Strong Bearish', 'Strong Selling Climax']):
            return True

        if position_type == 'SHORT' and any(term in market_pressure for term in
                                            ['Strong Buying', 'Strong Bullish', 'Strong Buying Climax']):
            return True

        return False

    def _adjust_position_size(self, base_size, volume_ratio, setup_quality):
        """Adjust position size based on volume and setup quality"""
        # Volume-based adjustment
        if volume_ratio >= 10.0:
            volume_mult = 1.5
        elif volume_ratio >= 5.0:
            volume_mult = 1.2
        elif volume_ratio >= 3.0:
            volume_mult = 1.0
        else:
            volume_mult = 0.7

        # Quality-based adjustment (0.8-1.2 range)
        quality_mult = 0.8 + (setup_quality / 100) * 0.4

        return base_size * volume_mult * quality_mult

    def _calculate_dynamic_take_profits(self, entry_price, stop_loss, side, setup_type, volume_ratio):
        """
        Calculate improved take profit levels based on setup type and volume ratio.

        Returns:
        --------
        Dictionary with optimized take profit levels
        """
        risk_distance = abs(entry_price - stop_loss)

        # Adjust TP ratios based on volume ratio - higher volume = higher targets
        # Cap at 2.0x for volume ratio of 10+
        volume_multiplier = 1.0 + (min(volume_ratio, 10) / 10)

        # Base TP ratios by setup type
        if 'REVERSAL' in setup_type or 'BOS' in setup_type:
            # Reversal setups often have larger moves
            base_ratios = {'tp1': 1.0, 'tp2': 2.0, 'tp3': 3.5}
        elif 'BREAKOUT' in setup_type or 'BREAKDOWN' in setup_type:
            # Breakouts can have strong momentum
            base_ratios = {'tp1': 0.8, 'tp2': 1.8, 'tp3': 3.0}
        elif 'CONTINUATION' in setup_type:
            # Continuation moves in established trends
            base_ratios = {'tp1': 1.0, 'tp2': 2.2, 'tp3': 3.2}
        elif 'CHoCH' in setup_type:
            # Change of character setups
            base_ratios = {'tp1': 1.2, 'tp2': 2.5, 'tp3': 4.0}
        else:
            # Default for other setup types
            base_ratios = {'tp1': 0.8, 'tp2': 1.5, 'tp3': 2.5}

        # Apply volume multiplier to TP ratios
        tp_ratios = {k: v * volume_multiplier for k, v in base_ratios.items()}

        # Calculate TP prices
        take_profits = {}
        if side == "LONG":
            for level, ratio in tp_ratios.items():
                take_profits[level] = entry_price + (risk_distance * ratio)
        else:  # SHORT
            for level, ratio in tp_ratios.items():
                take_profits[level] = entry_price - (risk_distance * ratio)

        return take_profits

    def backtest(self, data: pd.DataFrame, orders: List[Dict]) -> Dict:
        """
        Backtest the strategy on historical data.

        Parameters:
        -----------
        data: DataFrame with OHLCV data
        orders: List of order dictionaries from generate_orders

        Returns:
        --------
        Dict with backtest results
        """
        # Reset backtest state
        self.capital = self.initial_capital
        self.open_positions = []
        self.trade_history = []
        self.current_trades = {}

        # Create a copy of the data for results
        results = data.copy()
        results['capital'] = self.initial_capital
        results['equity'] = self.initial_capital
        results['open_positions'] = 0

        # Track PnL
        results['daily_pnl'] = 0.0
        results['cumulative_pnl'] = 0.0

        # Process each candle
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]

            # Check for order entries
            for order in orders:
                # Skip orders that are already processed or from the future
                if order['status'] != 'PENDING' or order['timestamp'] > current_time:
                    continue

                # Check if limit order is triggered
                if order['entry_type'] == 'LIMIT':
                    if (order['side'] == 'LONG' and low_price <= order['entry_price']) or \
                       (order['side'] == 'SHORT' and high_price >= order['entry_price']):
                        # Execute the order
                        order['status'] = 'ACTIVE'
                        order['entry_time'] = current_time
                        order['actual_entry_price'] = order['entry_price']
                        self.open_positions.append(order)
                        self.current_trades[id(order)] = order

                # Execute market orders immediately
                elif order['entry_type'] == 'MARKET':
                    order['status'] = 'ACTIVE'
                    order['entry_time'] = current_time
                    order['actual_entry_price'] = current_price
                    self.open_positions.append(order)
                    self.current_trades[id(order)] = order

            # Check for stop losses and take profits
            positions_to_close = []
            for position in self.open_positions:
                if position['status'] != 'ACTIVE':
                    continue

                # Check stop loss
                if (position['side'] == 'LONG' and low_price <= position['stop_loss']) or \
                   (position['side'] == 'SHORT' and high_price >= position['stop_loss']):
                    position['status'] = 'CLOSED'
                    position['exit_time'] = current_time
                    position['exit_price'] = position['stop_loss']
                    position['exit_reason'] = 'STOP_LOSS'
                    position['profit'] = self._calculate_profit(position)
                    self.capital += position['profit']
                    positions_to_close.append(position)
                    self.trade_history.append(position)
                    continue

                # Check take profits
                for tp_level, tp_price in position['take_profit'].items():
                    if (position['side'] == 'LONG' and high_price >= tp_price) or \
                       (position['side'] == 'SHORT' and low_price <= tp_price):
                        position['status'] = 'CLOSED'
                        position['exit_time'] = current_time
                        position['exit_price'] = tp_price
                        position['exit_reason'] = tp_level.upper()
                        position['profit'] = self._calculate_profit(position)
                        self.capital += position['profit']
                        positions_to_close.append(position)
                        self.trade_history.append(position)
                        break

            # Remove closed positions
            for position in positions_to_close:
                self.open_positions.remove(position)
                if id(position) in self.current_trades:
                    del self.current_trades[id(position)]

            # Calculate equity (capital + unrealized PnL)
            unrealized_pnl = sum(self._calculate_unrealized_profit(p, current_price)
                                 for p in self.open_positions)
            equity = self.capital + unrealized_pnl

            # Update results
            results.loc[current_time, 'capital'] = self.capital
            results.loc[current_time, 'equity'] = equity
            results.loc[current_time, 'open_positions'] = len(
                self.open_positions)

            # Calculate daily PnL
            if i > 0:
                results.loc[current_time, 'daily_pnl'] = results.loc[current_time, 'equity'] - \
                    results.loc[data.index[i-1], 'equity']

            # Calculate cumulative PnL
            results.loc[current_time, 'cumulative_pnl'] = results.loc[current_time, 'equity'] - \
                self.initial_capital

        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            results, self.trade_history)

        return {
            'results': results,
            'trades': self.trade_history,
            'performance': performance
        }

    def _calculate_profit(self, position: Dict) -> float:
        """Calculate profit/loss for a closed position"""
        entry_price = position['actual_entry_price']
        exit_price = position['exit_price']
        position_size = position['position_size']

        if position['side'] == 'LONG':
            return position_size * (exit_price - entry_price)
        else:  # SHORT
            return position_size * (entry_price - exit_price)

    def _calculate_unrealized_profit(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized profit/loss for an open position"""
        entry_price = position['actual_entry_price']
        position_size = position['position_size']

        if position['side'] == 'LONG':
            return position_size * (current_price - entry_price)
        else:  # SHORT
            return position_size * (entry_price - current_price)

    def _calculate_performance_metrics(self, results: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from backtest results"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return': 0,
                'annualized_return': 0
            }

        # Basic trade metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        losing_trades = sum(1 for t in trades if t['profit'] <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit metrics
        gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] <= 0))
        profit_factor = gross_profit / \
            gross_loss if gross_loss > 0 else float('inf')

        # Calculate drawdown
        equity_curve = results['equity']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        max_drawdown = drawdown.max()

        # Calculate returns
        total_return = (results['equity'].iloc[-1] -
                        self.initial_capital) / self.initial_capital

        # Calculate daily returns for Sharpe ratio
        daily_returns = results['daily_pnl'] / results['equity'].shift(1)
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * \
            np.sqrt(252) if daily_returns.std() > 0 else 0

        # Calculate annualized return
        days = (results.index[-1] - results.index[0]).days
        annualized_return = ((1 + total_return) **
                             (365 / max(days, 1))) - 1 if days > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'average_win': gross_profit / winning_trades if winning_trades > 0 else 0,
            'average_loss': gross_loss / losing_trades if losing_trades > 0 else 0,
            'largest_win': max([t['profit'] for t in trades if t['profit'] > 0], default=0),
            'largest_loss': min([t['profit'] for t in trades if t['profit'] <= 0], default=0),
            'avg_trade_duration': sum((t['exit_time'] - t['entry_time']).total_seconds()
                                      for t in trades) / total_trades / 3600 if total_trades > 0 else 0  # in hours
        }

    def _adjust_position_size_by_volume(self, base_position_size: float, volume_ratio: float) -> float:
        """
        Adjust position size based on volume ratio.
        Higher volume ratio = larger position size.

        Parameters:
        -----------
        base_position_size: Base position size calculated from risk
        volume_ratio: OB/Avg volume ratio

        Returns:
        --------
        Adjusted position size
        """
        # Define volume ratio thresholds and corresponding multipliers
        volume_tiers = {
            10.0: 1.5,   # Very high volume: increase position by 50%
            7.0: 1.3,    # High volume: increase position by 30%
            5.0: 1.2,    # Good volume: increase position by 20%
            3.0: 1.0,    # Adequate volume: standard position
            0.0: 0.7     # Low volume: reduce position by 30%
        }

        # Find the appropriate multiplier
        multiplier = 0.7  # Default to lowest tier
        for threshold, mult in sorted(volume_tiers.items()):
            if volume_ratio >= threshold:
                multiplier = mult
                break

        return base_position_size * multiplier

    def _score_setup_by_volume_and_pressure(self, setup: Dict, market_pressure: str) -> float:
        """
        Score a trading setup based on volume and market pressure.

        Parameters:
        -----------
        setup: Trading setup dictionary
        market_pressure: Current market pressure description

        Returns:
        --------
        Score adjustment (-20 to +20)
        """
        score_adjustment = 0
        volume_ratio = setup.get('ob_volume_ratio', 0)
        position_type = setup.get('position_type', '')

        # Volume ratio scoring
        if volume_ratio >= 10:
            score_adjustment += 20
        elif volume_ratio >= 7:
            score_adjustment += 15
        elif volume_ratio >= 5:
            score_adjustment += 10
        elif volume_ratio >= 3:
            score_adjustment += 5
        else:
            score_adjustment -= 10

        # Market pressure scoring
        is_bullish_pressure = any(
            term in market_pressure for term in ['Buying', 'Bullish'])
        is_bearish_pressure = any(term in market_pressure for term in [
                                  'Selling', 'Bearish'])
        is_strong_pressure = 'Strong' in market_pressure

        # Align pressure with position type
        if position_type == 'LONG':
            if is_bullish_pressure:
                score_adjustment += 10 if is_strong_pressure else 5
            elif is_bearish_pressure:
                score_adjustment -= 15 if is_strong_pressure else 5
        elif position_type == 'SHORT':
            if is_bearish_pressure:
                score_adjustment += 10 if is_strong_pressure else 5
            elif is_bullish_pressure:
                score_adjustment -= 15 if is_strong_pressure else 5

        return score_adjustment

    def _has_critical_warnings(self, setup: Dict) -> bool:
        """
        Check if setup has critical warning messages that should prevent order generation.

        Returns:
        - True if critical warnings are present, False otherwise
        """
        warning_messages = setup.get('warning_messages', [])

        critical_warnings = [
            "Strong bearish momentum against bullish setup",
            "Strong bullish momentum against bearish setup",
            "Low volume confidence",
            "Insufficient volume for reliable setup",
            "Extreme market volatility detected",
            "Contradictory signals detected",
            "Setup too far from current price"
        ]

        # Check if any critical warning is in the warning messages
        for critical in critical_warnings:
            for warning in warning_messages:
                if critical.lower() in warning.lower():
                    return True

        return False


class LiveTradingExecutor:
    """
    Executes trades on a live exchange based on the FuturesStrategy signals.
    """

    def __init__(self, exchange_client, strategy: FuturesStrategy):
        """
        Initialize the live trading executor.

        Parameters:
        -----------
        exchange_client: Exchange API client (e.g., Binance client)
        strategy: FuturesStrategy instance
        """
        self.exchange = exchange_client
        self.strategy = strategy
        self.active_orders = {}

    def execute_orders(self, orders: List[Dict]) -> List[Dict]:
        """
        Execute orders on the exchange.

        Parameters:
        -----------
        orders: List of order dictionaries from strategy.generate_orders()

        Returns:
        --------
        List of executed order details
        """
        executed_orders = []

        for order in orders:
            try:
                # Set leverage first
                self.exchange.futures_change_leverage(
                    symbol=order['symbol'],
                    leverage=order['leverage']
                )

                # Calculate quantity in exchange format
                quantity = order['position_size']

                # Determine order side for exchange
                side = "BUY" if order['side'] == "LONG" else "SELL"

                # Execute the order based on entry type
                if order['entry_type'] == "LIMIT":
                    # Place limit order
                    response = self.exchange.futures_create_order(
                        symbol=order['symbol'],
                        side=side,
                        type='LIMIT',
                        timeInForce='GTC',
                        quantity=quantity,
                        price=order['entry_price']
                    )

                    # Store the order ID for tracking
                    order['exchange_order_id'] = response['orderId']
                    order['status'] = 'PENDING'

                else:  # MARKET order
                    # Place market order
                    response = self.exchange.futures_create_order(
                        symbol=order['symbol'],
                        side=side,
                        type='MARKET',
                        quantity=quantity
                    )

                    # Get fill price
                    fill_price = float(response['avgPrice'])
                    order['actual_entry_price'] = fill_price
                    order['exchange_order_id'] = response['orderId']
                    order['status'] = 'ACTIVE'

                    # Place stop loss order
                    sl_response = self.exchange.futures_create_order(
                        symbol=order['symbol'],
                        side="SELL" if side == "BUY" else "BUY",
                        type='STOP_MARKET',
                        quantity=quantity,
                        stopPrice=order['stop_loss']
                    )

                    order['sl_order_id'] = sl_response['orderId']

                    # Place take profit orders (can be implemented with OCO orders)
                    # This is a simplified version - in practice you might want to split the position
                    tp_response = self.exchange.futures_create_order(
                        symbol=order['symbol'],
                        side="SELL" if side == "BUY" else "BUY",
                        type='TAKE_PROFIT_MARKET',
                        quantity=quantity,
                        # Using TP2 as default
                        stopPrice=order['take_profit']['tp2']
                    )

                    order['tp_order_id'] = tp_response['orderId']

                executed_orders.append(order)
                self.active_orders[order['exchange_order_id']] = order

                logger.info(
                    f"Executed {order['side']} order for {order['symbol']} at {order.get('actual_entry_price', order['entry_price'])}")

            except Exception as e:
                logger.error(f"Error executing order: {str(e)}")
                order['status'] = 'ERROR'
                order['error'] = str(e)
                executed_orders.append(order)

        return executed_orders

    def check_order_status(self) -> List[Dict]:
        """
        Check the status of pending orders and update accordingly.

        Returns:
        --------
        List of updated orders
        """
        updated_orders = []

        for order_id, order in list(self.active_orders.items()):
            try:
                # Skip orders that are already active or closed
                if order['status'] not in ['PENDING', 'ACTIVE']:
                    continue

                # Check order status from exchange
                order_status = self.exchange.futures_get_order(
                    symbol=order['symbol'],
                    orderId=order_id
                )

                # Update order status
                if order_status['status'] == 'FILLED' and order['status'] == 'PENDING':
                    order['status'] = 'ACTIVE'
                    order['actual_entry_price'] = float(
                        order_status['avgPrice'])
                    order['entry_time'] = datetime.fromtimestamp(
                        order_status['updateTime'] / 1000)

                    # Place stop loss and take profit orders
                    side = "BUY" if order['side'] == "SHORT" else "SELL"
                    quantity = order['position_size']

                    # Place stop loss
                    sl_response = self.exchange.futures_create_order(
                        symbol=order['symbol'],
                        side=side,
                        type='STOP_MARKET',
                        quantity=quantity,
                        stopPrice=order['stop_loss']
                    )

                    order['sl_order_id'] = sl_response['orderId']

                    # Place take profit (simplified)
                    tp_response = self.exchange.futures_create_order(
                        symbol=order['symbol'],
                        side=side,
                        type='TAKE_PROFIT_MARKET',
                        quantity=quantity,
                        stopPrice=order['take_profit']['tp2']
                    )

                    order['tp_order_id'] = tp_response['orderId']

                    updated_orders.append(order)
                    logger.info(
                        f"Order {order_id} for {order['symbol']} filled at {order['actual_entry_price']}")

                elif order_status['status'] == 'CANCELED':
                    order['status'] = 'CANCELED'
                    del self.active_orders[order_id]
                    updated_orders.append(order)
                    logger.info(
                        f"Order {order_id} for {order['symbol']} was canceled")

                # Check if stop loss or take profit was hit for active orders
                if order['status'] == 'ACTIVE' and ('sl_order_id' in order or 'tp_order_id' in order):
                    # Check stop loss
                    if 'sl_order_id' in order:
                        sl_status = self.exchange.futures_get_order(
                            symbol=order['symbol'],
                            orderId=order['sl_order_id']
                        )

                        if sl_status['status'] == 'FILLED':
                            order['status'] = 'CLOSED'
                            order['exit_price'] = float(sl_status['avgPrice'])
                            order['exit_time'] = datetime.fromtimestamp(
                                sl_status['updateTime'] / 1000)
                            order['exit_reason'] = 'STOP_LOSS'
                            order['profit'] = self._calculate_profit(order)

                            # Cancel take profit order
                            if 'tp_order_id' in order:
                                try:
                                    self.exchange.futures_cancel_order(
                                        symbol=order['symbol'],
                                        orderId=order['tp_order_id']
                                    )
                                except:
                                    pass

                            del self.active_orders[order_id]
                            updated_orders.append(order)
                            logger.info(
                                f"Stop loss triggered for {order['symbol']} at {order['exit_price']}")

                    # Check take profit
                    if 'tp_order_id' in order and order['status'] == 'ACTIVE':
                        tp_status = self.exchange.futures_get_order(
                            symbol=order['symbol'],
                            orderId=order['tp_order_id']
                        )

                        if tp_status['status'] == 'FILLED':
                            order['status'] = 'CLOSED'
                            order['exit_price'] = float(tp_status['avgPrice'])
                            order['exit_time'] = datetime.fromtimestamp(
                                tp_status['updateTime'] / 1000)
                            order['exit_reason'] = 'TAKE_PROFIT'
                            order['profit'] = self._calculate_profit(order)

                            # Cancel stop loss order
                            if 'sl_order_id' in order:
                                try:
                                    self.exchange.futures_cancel_order(
                                        symbol=order['symbol'],
                                        orderId=order['sl_order_id']
                                    )
                                except:
                                    pass

                            del self.active_orders[order_id]
                            updated_orders.append(order)
                            logger.info(
                                f"Take profit triggered for {order['symbol']} at {order['exit_price']}")

            except Exception as e:
                logger.error(f"Error checking order status: {str(e)}")

        return updated_orders

    def _calculate_profit(self, order: Dict) -> float:
        """Calculate profit/loss for a closed position"""
        entry_price = order['actual_entry_price']
        exit_price = order['exit_price']
        position_size = order['position_size']

        if order['side'] == 'LONG':
            return position_size * (exit_price - entry_price)
        else:  # SHORT
            return position_size * (entry_price - exit_price)

    def cancel_all_orders(self, symbol: str = None) -> None:
        """
        Cancel all active orders for a symbol or all symbols.

        Parameters:
        -----------
        symbol: Optional symbol to cancel orders for. If None, cancels all orders.
        """
        try:
            if symbol:
                self.exchange.futures_cancel_all_open_orders(symbol=symbol)
                logger.info(f"Canceled all orders for {symbol}")
            else:
                # Get all symbols with active orders
                symbols = set(order['symbol']
                              for order in self.active_orders.values())
                for sym in symbols:
                    self.exchange.futures_cancel_all_open_orders(symbol=sym)
                logger.info("Canceled all orders across all symbols")

            # Update local order status
            for order_id, order in list(self.active_orders.items()):
                if symbol is None or order['symbol'] == symbol:
                    order['status'] = 'CANCELED'
                    del self.active_orders[order_id]

        except Exception as e:
            logger.error(f"Error canceling orders: {str(e)}")
