import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from binance_data_fetcher import BinanceDataFetcher
from futures_strategy import FuturesStrategy
from smartmoneyconcepts.smc import smc
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Backtester for the Futures Strategy with detailed performance tracking
    and visualization capabilities.
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_risk_per_trade: float = 0.02,
                 leverage: int = 20,
                 symbol: str = "BTCUSDT"):
        """
        Initialize the strategy backtester.

        Parameters:
        -----------
        initial_capital: Starting capital for backtesting
        max_risk_per_trade: Maximum risk per trade as a percentage of capital
        leverage: Fixed leverage to use for all trades
        symbol: Trading symbol to backtest
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.leverage = leverage
        self.symbol = symbol

        # Initialize strategy
        self.strategy = FuturesStrategy(
            initial_capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade,
            default_leverage=leverage
        )

        # Initialize data fetcher
        self.data_fetcher = BinanceDataFetcher()

        # Track performance
        self.equity_curve = []
        self.trade_history = []
        self.drawdowns = []
        self.monthly_returns = {}

        # Results storage
        self.results = None

    def fetch_historical_data(self,
                              interval: str = "1h",
                              start_date: str = None,
                              end_date: str = None,
                              limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical data for backtesting.

        Parameters:
        -----------
        interval: Timeframe interval (e.g., "1h", "4h", "1d")
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        limit: Maximum number of candles to fetch

        Returns:
        --------
        DataFrame with historical price data
        """
        logger.info(f"Fetching historical data for {self.symbol} {interval}")

        # Fetch data
        data = self.data_fetcher.get_historical_klines(
            symbol=self.symbol,
            interval=interval,
            start_time=start_date,
            limit=limit
        )

        logger.info(
            f"Fetched {len(data)} candles from {data.index[0]} to {data.index[-1]}")
        return data

    def _rebalance_orders(self, new_orders: List[Dict], current_time: datetime):
        """
        Rebalance orders by prioritizing the best setups when capital is limited.

        Parameters:
        -----------
        new_orders: List of new orders to consider
        current_time: Current timestamp for logging

        Returns:
        --------
        List of orders to process after rebalancing
        """
        # Get all pending orders (existing + new)
        existing_pending_orders = [order for order in self.strategy.pending_orders
                                   if order['status'] == 'PENDING']
        all_pending_orders = existing_pending_orders + new_orders

        # If no orders, just return empty list
        if not all_pending_orders:
            return []

        # Calculate total margin required
        total_margin_required = sum(
            [order.get('margin_amount', 0) for order in all_pending_orders])

        # If we have enough capital for all orders, return all of them
        if total_margin_required <= self.capital:
            return all_pending_orders

        # We need to prioritize orders
        logger.info(
            f"Rebalancing orders. Required: ${total_margin_required:.2f}, Available: ${self.capital:.2f}")

        # Score each order based on multiple factors
        for order in all_pending_orders:
            # Base score is setup quality
            score = order.get('setup_quality', 0)

            # Adjust score based on volume ratio (higher volume = higher score)
            volume_ratio = order.get('volume_ratio', 1)
            score += min(volume_ratio * 2, 20)  # Cap volume bonus at 20 points

            # Adjust score based on risk-reward ratio (higher R:R = higher score)
            rr_ratio = order.get('risk_reward_ratio', {}).get('tp2', 1)
            score += min(rr_ratio * 5, 25)  # Cap R:R bonus at 25 points

            # Adjust score based on proximity to entry (closer = higher score)
            if 'price_to_entry_percent' in order:
                proximity = abs(order['price_to_entry_percent'])
                if proximity < 0.5:  # Very close to entry
                    score += 15
                elif proximity < 1.0:  # Close to entry
                    score += 10
                elif proximity < 2.0:  # Moderately close
                    score += 5

            # Penalize orders that have been pending for too long
            if 'created_time' in order:
                hours_pending = (
                    current_time - order['created_time']).total_seconds() / 3600
                if hours_pending > 24:
                    score -= 20  # Heavy penalty for orders pending > 24 hours
                elif hours_pending > 12:
                    score -= 10  # Moderate penalty for orders pending > 12 hours

            # Store the score
            order['rebalance_score'] = score

        # Sort orders by score (highest first)
        all_pending_orders.sort(key=lambda x: x.get(
            'rebalance_score', 0), reverse=True)

        # Try different approaches to maximize capital utilization

        # Approach 1: Greedy algorithm - take highest scoring orders first
        greedy_orders = []
        remaining_capital = self.capital

        for order in all_pending_orders:
            margin_required = order.get('margin_amount', 0)
            if margin_required <= remaining_capital:
                greedy_orders.append(order)
                remaining_capital -= margin_required

        # Approach 2: Try to fit smaller orders if there's still capital left
        if remaining_capital > 100:  # If we have at least $100 left
            # Sort remaining orders by margin amount (smallest first)
            remaining_orders = [
                o for o in all_pending_orders if o not in greedy_orders]
            remaining_orders.sort(key=lambda x: x.get(
                'margin_amount', float('inf')))

            for order in remaining_orders:
                margin_required = order.get('margin_amount', 0)
                if margin_required <= remaining_capital:
                    greedy_orders.append(order)
                    remaining_capital -= margin_required

        # Approach 3: If we couldn't fit any orders, take the best single order we can afford
        if not greedy_orders and all_pending_orders:
            # Find the highest scoring order we can afford
            affordable_orders = [o for o in all_pending_orders
                                 if o.get('margin_amount', float('inf')) <= self.capital]

            if affordable_orders:
                # Sort by score and take the best one
                affordable_orders.sort(key=lambda x: x.get(
                    'rebalance_score', 0), reverse=True)
                greedy_orders = [affordable_orders[0]]
            else:
                # If no order is affordable, try to adjust the best order to fit our capital
                # Copy to avoid modifying original
                best_order = all_pending_orders[0].copy()

                # Adjust margin to match available capital
                original_margin = best_order.get('margin_amount', 0)
                if original_margin > 0:
                    scaling_factor = self.capital / original_margin

                    # Only proceed if we can allocate at least 50% of the original margin
                    if scaling_factor >= 0.5:
                        best_order['margin_amount'] = self.capital
                        best_order['position_size'] = best_order.get(
                            'position_size', 0) * scaling_factor
                        best_order['max_loss_amount'] = best_order.get(
                            'max_loss_amount', 0) * scaling_factor

                        logger.info(
                            f"Adjusted order margin from ${original_margin:.2f} to ${self.capital:.2f}")
                        greedy_orders = [best_order]

        # Mark orders that weren't selected as canceled
        for order in all_pending_orders:
            if order in existing_pending_orders and order not in greedy_orders:
                order['status'] = 'CANCELED'
                order['cancel_reason'] = 'REBALANCED_FOR_BETTER_SETUP'

        # Log rebalancing results
        canceled_count = len(existing_pending_orders) - \
            sum(1 for o in greedy_orders if o in existing_pending_orders)
        new_accepted = sum(1 for o in greedy_orders if o in new_orders)

        total_allocated = sum([order.get('margin_amount', 0)
                              for order in greedy_orders])
        utilization = (total_allocated / self.capital) * \
            100 if self.capital > 0 else 0

        logger.info(
            f"Rebalancing complete: Canceled {canceled_count} existing orders, accepted {new_accepted}/{len(new_orders)} new orders")
        logger.info(
            f"Capital utilization: ${total_allocated:.2f}/{self.capital:.2f} ({utilization:.1f}%)")

        return greedy_orders

    def run_backtest(self,
                     data: pd.DataFrame,
                     window_size: int = 100,
                     min_setup_quality: float = 70.0,
                     min_volume_ratio: float = 3.0) -> Dict:
        """
        Run backtest on historical data with improved order rebalancing.

        Parameters:
        -----------
        data: DataFrame with historical price data
        window_size: Number of candles to use for each analysis window
        min_setup_quality: Minimum setup quality to consider
        min_volume_ratio: Minimum volume ratio to consider

        Returns:
        --------
        Dictionary with backtest results
        """
        logger.info(f"Running backtest with {len(data)} candles")

        # Reset state
        self.capital = self.initial_capital
        self.equity_curve = []
        self.trade_history = []
        self.drawdowns = []
        self.monthly_returns = {}
        self.strategy.open_positions = []
        self.strategy.pending_orders = []

        # Prepare results dataframe
        results = pd.DataFrame(index=data.index)
        results['capital'] = self.initial_capital
        results['equity'] = self.initial_capital
        results['drawdown'] = 0.0
        results['open_positions'] = 0

        # Ensure we have enough data for the initial window
        if len(data) < window_size:
            logger.error(
                f"Not enough data for window size {window_size}. Data length: {len(data)}")
            return {}

        # Process data sequentially with proper windowing
        for i in range(0, len(data) - window_size):
            # Analysis window is fixed size, ending at current position + window_size
            analysis_end = i + window_size
            current_time = data.index[analysis_end]

            # Analysis window includes data up to but not including the current candle
            analysis_window = data.iloc[i:analysis_end].copy()

            # Current candle is the one we're processing for order execution
            current_candle = data.iloc[analysis_end]
            current_price = current_candle['close']

            # Analyze market using the analysis window
            analysis = self.strategy.analyze_market(
                analysis_window, ignore_old_ob=False)

            # Generate new orders
            new_orders = self.strategy.generate_orders(
                analysis=analysis,
                min_setup_quality=min_setup_quality,
                min_volume_ratio=min_volume_ratio,
                respect_pressure=True,
                respect_warnings=True
            )

            # Add timestamp and initial status to new orders
            for order in new_orders:
                order['created_time'] = current_time
                order['status'] = 'PENDING'
                order['price_to_entry_percent'] = (
                    (order['entry_price'] / current_price) - 1) * 100

            # Rebalance orders (existing + new)
            rebalanced_orders = self._rebalance_orders(
                new_orders, current_time)

            # Update strategy's pending orders
            self.strategy.pending_orders = [order for order in rebalanced_orders
                                            if order['status'] == 'PENDING']

            # Process the current candle
            self._process_candle(current_candle, self.strategy.pending_orders)

            # Calculate equity (capital + unrealized P&L)
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            equity = self.capital + unrealized_pnl

            # Update equity curve
            self.equity_curve.append((current_time, equity))

            # Calculate drawdown
            peak_equity = max([eq for _, eq in self.equity_curve])
            drawdown = (peak_equity - equity) / \
                peak_equity if peak_equity > 0 else 0
            self.drawdowns.append((current_time, drawdown))

            # Update monthly returns
            month_key = current_time.strftime('%Y-%m')
            if month_key not in self.monthly_returns:
                # First observation for this month
                self.monthly_returns[month_key] = {
                    'start_equity': equity,
                    'end_equity': equity
                }
            else:
                # Update end equity for this month
                self.monthly_returns[month_key]['end_equity'] = equity

            # Update results dataframe
            results.loc[current_time, 'capital'] = self.capital
            results.loc[current_time, 'equity'] = equity
            results.loc[current_time, 'drawdown'] = drawdown
            results.loc[current_time, 'open_positions'] = len(
                self.strategy.open_positions)

        # Calculate performance metrics
        performance = self._calculate_performance_metrics()

        # Store results
        self.results = {
            'equity_curve': self.equity_curve,
            'trade_history': self.trade_history,
            'drawdowns': self.drawdowns,
            'monthly_returns': self.monthly_returns,
            'performance': performance
        }

        logger.info(
            f"Backtest completed with {len(self.trade_history)} trades")
        logger.info(
            f"Final capital: ${self.capital:.2f}, Return: {performance['total_return']:.2%}")

        return self.results

    def _process_candle(self, candle: pd.Series, orders: List[Dict]):
        """Process a candle and execute any orders that should be triggered"""
        high = candle['high']
        low = candle['low']
        close = candle['close']

        # Track positions that need to be closed
        positions_to_close = []

        # Process pending orders
        pending_orders = [
            order for order in orders if order['status'] == 'PENDING']

        # Process the orders
        for order in pending_orders:
            # Check if order should be executed
            if order['entry_type'] == 'MARKET':
                # Market orders execute immediately
                self._execute_entry(order, close)
            elif order['entry_type'] == 'LIMIT':
                # Limit orders execute if price reaches the limit price
                if (order['side'] == 'LONG' and low <= order['entry_price']) or \
                   (order['side'] == 'SHORT' and high >= order['entry_price']):
                    self._execute_entry(order, order['entry_price'])

        # Check stop loss and take profit for open positions
        for position in self.strategy.open_positions:
            if position['status'] != 'ACTIVE':
                continue

            # Check stop loss
            if (position['side'] == 'LONG' and low <= position['stop_loss']) or \
               (position['side'] == 'SHORT' and high >= position['stop_loss']):
                self._execute_exit(
                    position, position['stop_loss'], 'STOP_LOSS')
                positions_to_close.append(position)
                continue

            # Check take profits
            for tp_level, tp_price in position['take_profit'].items():
                if (position['side'] == 'LONG' and high >= tp_price) or \
                   (position['side'] == 'SHORT' and low <= tp_price):
                    self._execute_exit(position, tp_price, tp_level.upper())
                    positions_to_close.append(position)
                    break

        # Remove closed positions
        for position in positions_to_close:
            if position in self.strategy.open_positions:
                self.strategy.open_positions.remove(position)

    def _execute_entry(self, order: Dict, price: float):
        """Execute an entry order"""
        # Calculate margin required
        margin_required = order.get('margin_amount', 0)

        # Check if we have enough capital
        if margin_required > self.capital:
            logger.warning(
                f"Insufficient capital for order. Required: ${margin_required:.2f}, Available: ${self.capital:.2f}")
            order['status'] = 'CANCELED'
            order['cancel_reason'] = 'INSUFFICIENT_MARGIN'
            return

        # Deduct margin from capital
        self.capital -= margin_required

        # Update order status
        order['status'] = 'ACTIVE'
        order['entry_time'] = datetime.now()
        order['actual_entry_price'] = price
        order['margin_used'] = margin_required

        # Add to open positions
        self.strategy.open_positions.append(order)

        logger.debug(f"Executed {order['side']} entry at {price}")

    def _execute_exit(self, position: Dict, price: float, reason: str):
        """Execute an exit order"""
        position['status'] = 'CLOSED'
        position['exit_time'] = datetime.now()
        position['exit_price'] = price
        position['exit_reason'] = reason

        # Calculate profit/loss
        entry_price = position['actual_entry_price']
        position_size = position['position_size']

        if position['side'] == 'LONG':
            profit = position_size * (price - entry_price) / entry_price
        else:  # SHORT
            profit = position_size * (entry_price - price) / entry_price

        position['profit'] = profit
        position['profit_percent'] = (
            profit / position.get('margin_used', 1)) * 100

        # Return margin to capital
        margin_used = position.get('margin_used', 0)
        self.capital += margin_used

        # Add profit/loss to capital
        self.capital += profit

        # Add to trade history
        self.trade_history.append(position)

        logger.debug(
            f"Executed {position['side']} exit at {price}, reason: {reason}, profit: ${profit:.2f}")

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss of open positions"""
        unrealized_pnl = 0.0

        for position in self.strategy.open_positions:
            if position['status'] != 'ACTIVE':
                continue

            entry_price = position['actual_entry_price']
            position_size = position['position_size']

            if position['side'] == 'LONG':
                unrealized_pnl += position_size * \
                    (current_price - entry_price) / entry_price
            else:  # SHORT
                unrealized_pnl += position_size * \
                    (entry_price - current_price) / entry_price

        return unrealized_pnl

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return': 0,
                'average_win': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'annualized_return': 0
            }

        # Basic trade metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['profit'] > 0)
        losing_trades = sum(1 for t in self.trade_history if t['profit'] <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit metrics
        gross_profit = sum(t['profit']
                           for t in self.trade_history if t['profit'] > 0)
        gross_loss = abs(sum(t['profit']
                         for t in self.trade_history if t['profit'] <= 0))
        profit_factor = gross_profit / \
            gross_loss if gross_loss > 0 else float('inf')

        # Calculate max drawdown
        max_drawdown = max([dd for _, dd in self.drawdowns]
                           ) if self.drawdowns else 0

        # Calculate returns
        if self.equity_curve:
            final_equity = self.equity_curve[-1][1]
            total_return = (final_equity - self.initial_capital) / \
                self.initial_capital
        else:
            total_return = 0

        # Calculate monthly returns
        monthly_return_values = []
        for month, values in self.monthly_returns.items():
            start_equity = values['start_equity']
            end_equity = values['end_equity']
            monthly_return = (end_equity - start_equity) / \
                start_equity if start_equity > 0 else 0
            monthly_return_values.append(monthly_return)

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if monthly_return_values:
            avg_monthly_return = np.mean(monthly_return_values)
            std_monthly_return = np.std(monthly_return_values)
            sharpe_ratio = avg_monthly_return / std_monthly_return * \
                np.sqrt(12) if std_monthly_return > 0 else 0

            # Annualized return
            annualized_return = (1 + avg_monthly_return) ** 12 - 1
        else:
            sharpe_ratio = 0
            annualized_return = 0

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
            'largest_win': max([t['profit'] for t in self.trade_history if t['profit'] > 0], default=0),
            'largest_loss': min([t['profit'] for t in self.trade_history if t['profit'] <= 0], default=0),
            'monthly_returns': {month: (values['end_equity'] - values['start_equity']) / values['start_equity']
                                for month, values in self.monthly_returns.items()}
        }

    def plot_results(self, save_path: str = None):
        """
        Plot backtest results with multiple subplots.

        Parameters:
        -----------
        save_path: Path to save the plot image (optional)
        """
        if self.results is None:
            logger.error("No backtest results to plot")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))

        # Define grid for subplots
        gs = fig.add_gridspec(3, 2)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        dates = [date for date, _ in self.equity_curve]
        equity = [eq for _, eq in self.equity_curve]
        ax1.plot(dates, equity, label='Equity Curve', color='blue')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True)

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        drawdown_dates = [date for date, _ in self.drawdowns]
        # Convert to percentage
        drawdown_values = [dd * 100 for _, dd in self.drawdowns]
        ax2.fill_between(drawdown_dates, drawdown_values,
                         0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)

        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. Monthly returns
        ax3 = fig.add_subplot(gs[1, 1])
        months = list(self.monthly_returns.keys())
        returns = [(self.monthly_returns[m]['end_equity'] - self.monthly_returns[m]['start_equity']) /
                   self.monthly_returns[m]['start_equity'] * 100 for m in months]

        colors = ['green' if r >= 0 else 'red' for r in returns]
        ax3.bar(months, returns, color=colors)
        ax3.set_title('Monthly Returns')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # 4. Trade distribution
        ax4 = fig.add_subplot(gs[2, 0])
        profits = [t['profit'] for t in self.trade_history]
        ax4.hist(profits, bins=20, color='blue', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--')
        ax4.set_title('Trade Profit Distribution')
        ax4.set_xlabel('Profit ($)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)

        # 5. Performance metrics table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        performance = self.results['performance']
        print(performance)
        metrics_text = [
            f"Total Trades: {performance['total_trades']}",
            f"Win Rate: {performance['win_rate']:.2%}",
            f"Profit Factor: {performance['profit_factor']:.2f}",
            f"Max Drawdown: {performance['max_drawdown']:.2%}",
            f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}",
            f"Total Return: {performance['total_return']:.2%}",
            f"Annualized Return: {performance['annualized_return']:.2%}",
            f"Average Win: ${performance['average_win']:.2f}",
            f"Average Loss: ${performance['average_loss']:.2f}",
            f"Largest Win: ${performance['largest_win']:.2f}",
            f"Largest Loss: ${performance['largest_loss']:.2f}"
        ]

        ax5.text(0.1, 0.5, '\n'.join(metrics_text), fontsize=12,
                 verticalalignment='center')
        ax5.set_title('Performance Metrics')

        # Adjust layout
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def save_results(self, filepath: str):
        """
        Save backtest results to a file.

        Parameters:
        -----------
        filepath: Path to save the results
        """
        if self.results is None:
            logger.error("No backtest results to save")
            return

        # Convert results to serializable format
        serializable_results = {
            'equity_curve': [(date.strftime('%Y-%m-%d %H:%M:%S'), eq) for date, eq in self.equity_curve],
            'drawdowns': [(date.strftime('%Y-%m-%d %H:%M:%S'), dd) for date, dd in self.drawdowns],
            'monthly_returns': self.monthly_returns,
            'performance': self.results['performance'],
            'trade_history': []
        }

        # Convert trade history
        for trade in self.trade_history:
            serializable_trade = trade.copy()
        # Convert datetime objects to strings
            if isinstance(trade['entry_time'], datetime):
                serializable_trade['entry_time'] = trade['entry_time'].strftime(
                    '%Y-%m-%d %H:%M:%S')
            if isinstance(trade['exit_time'], datetime):
                serializable_trade['exit_time'] = trade['exit_time'].strftime(
                    '%Y-%m-%d %H:%M:%S')

        # Check for any other datetime objects in the trade dictionary
            for key, value in serializable_trade.items():
                if isinstance(value, datetime):
                    serializable_trade[key] = value.strftime(
                        '%Y-%m-%d %H:%M:%S')

            serializable_results['trade_history'].append(serializable_trade)

    # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)

        logger.info(f"Results saved to {filepath}")

    def load_results(self, filepath: str):
        """
        Load backtest results from a file.

        Parameters:
        -----------
        filepath: Path to load the results from
        """
        with open(filepath, 'r') as f:
            loaded_results = json.load(f)

        # Convert back to proper format
        self.equity_curve = [(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'), eq)
                             for date, eq in loaded_results['equity_curve']]

        self.drawdowns = [(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'), dd)
                          for date, dd in loaded_results['drawdowns']]

        self.monthly_returns = loaded_results['monthly_returns']

        # Convert trade history
        self.trade_history = []
        for trade in loaded_results['trade_history']:
            trade['entry_time'] = datetime.strptime(
                trade['entry_time'], '%Y-%m-%d %H:%M:%S')
            trade['exit_time'] = datetime.strptime(
                trade['exit_time'], '%Y-%m-%d %H:%M:%S')
            self.trade_history.append(trade)

        # Recreate results
        self.results = {
            'equity_curve': self.equity_curve,
            'trade_history': self.trade_history,
            'drawdowns': self.drawdowns,
            'monthly_returns': self.monthly_returns,
            'performance': loaded_results['performance']
        }

        logger.info(f"Results loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Create backtester
    backtester = StrategyBacktester(
        initial_capital=1000.0,
        max_risk_per_trade=0.02,
        leverage=20,
        symbol="BTCUSDT"
    )

    start_time = datetime.now() - timedelta(days=30)

    # Fetch historical data
    data = backtester.fetch_historical_data(
        interval="15m",
        start_date=start_time,
        limit=1000
    )

    # Run backtest
    results = backtester.run_backtest(
        data=data,
        window_size=300,
        min_setup_quality=70,
        min_volume_ratio=2.0
    )

    # Plot results
    backtester.plot_results(save_path="backtest_results.png")

    # Save results
    backtester.save_results("backtest_results.json")

    # Print summary
    print("\nBacktest Summary:")
    print(f"Symbol: {backtester.symbol}")
    print(f"Period: {data.index[0]} to {data.index[-1]}")
    print(f"Initial Capital: ${backtester.initial_capital:.2f}")
    print(f"Final Capital: ${backtester.capital:.2f}")
    print(f"Total Return: {results['performance']['total_return']:.2%}")
    print(f"Win Rate: {results['performance']['win_rate']:.2%}")
    print(f"Profit Factor: {results['performance']['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['performance']['total_trades']}")
