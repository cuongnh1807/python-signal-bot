import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from binance.client import Client
from typing import List, Dict, Optional, Tuple

from binance_data_fetcher import BinanceDataFetcher


class FairValueGap:
    """Class to represent Fair Value Gap information"""

    def __init__(self, top: float, bottom: float, is_bullish: bool, start_time: datetime, start_index: int):
        self.top = top
        self.bottom = bottom
        self.is_bullish = is_bullish  # True for bullish FVG, False for bearish
        self.start_time = start_time
        self.start_index = start_index
        self.mitigated = False
        self.mitigation_time = None
        self.mitigation_index = None
        self.mitigation_price = None

    def get_height(self) -> float:
        """Get the height of the FVG"""
        return abs(self.top - self.bottom)

    def get_midpoint(self) -> float:
        """Get the midpoint price of the FVG"""
        return (self.top + self.bottom) / 2

    def __str__(self):
        fvg_type = "Bull" if self.is_bullish else "Bear"
        status = "MITIGATED" if self.mitigated else "ACTIVE"
        return (f"{fvg_type} FVG [{status}]: {self.bottom:.4f} - {self.top:.4f} "
                f"at {self.start_time} (Height: {self.get_height():.6f})")


def calculate_threshold(df: pd.DataFrame, auto_threshold: bool = True, manual_threshold_pct: float = 0.0) -> pd.Series:
    """
    Calculate threshold for FVG detection

    Pine Script equivalent:
    threshold = auto ? ta.cum((high - low) / low) / bar_index : thresholdPer / 100
    """
    if auto_threshold:
        # Automatic threshold: cumulative average of relative ranges
        relative_ranges = (df['high'] - df['low']) / df['low']
        cumulative_sum = relative_ranges.cumsum()
        bar_indices = np.arange(1, len(df) + 1)
        threshold = cumulative_sum / bar_indices
    else:
        # Manual threshold
        threshold = pd.Series([manual_threshold_pct / 100]
                              * len(df), index=df.index)

    return threshold


def detect_fair_value_gaps(df: pd.DataFrame,
                           threshold_pct: float = 0.1,  # 0.1% default for noise filtering
                           min_height_pct: float = 0.08,  # 0.08% min box height
                           auto_threshold: bool = False,
                           timeframe: str = "chart") -> list:
    """
    Detect Fair Value Gaps using LuxAlgo logic, with noise filtering and min box height.
    """
    original_index = df.index.copy()
    df_reset = df.reset_index(drop=True)

    # Calculate threshold
    if auto_threshold:
        relative_ranges = (df_reset['high'] -
                           df_reset['low']) / df_reset['low']
        cumulative_sum = relative_ranges.cumsum()
        bar_indices = np.arange(1, len(df_reset) + 1)
        threshold_series = cumulative_sum / bar_indices
    else:
        threshold_series = pd.Series([threshold_pct / 100] * len(df_reset))

    fvg_list = []

    for i in range(2, len(df_reset)):
        current_high = df_reset.loc[i, 'high']
        current_low = df_reset.loc[i, 'low']
        prev_close = df_reset.loc[i-1, 'close']
        prev2_high = df_reset.loc[i-2, 'high']
        prev2_low = df_reset.loc[i-2, 'low']
        current_threshold = threshold_series.iloc[i]

        # Bullish FVG
        if (current_low > prev2_high and prev_close > prev2_high):
            gap_size = (current_low - prev2_high) / prev2_high
            height_pct = gap_size * 100
            if gap_size > current_threshold and height_pct > min_height_pct:
                fvg = FairValueGap(
                    top=current_low,
                    bottom=prev2_high,
                    is_bullish=True,
                    start_time=original_index[i],
                    start_index=i
                )
                fvg_list.append(fvg)
        # Bearish FVG
        elif (current_high < prev2_low and prev_close < prev2_low):
            gap_size = (prev2_low - current_high) / current_high
            height_pct = gap_size * 100
            if gap_size > current_threshold and height_pct > min_height_pct:
                fvg = FairValueGap(
                    top=prev2_low,
                    bottom=current_high,
                    is_bullish=False,
                    start_time=original_index[i],
                    start_index=i
                )
                fvg_list.append(fvg)
    return fvg_list


def check_fvg_mitigation(fvg_list: List[FairValueGap], df: pd.DataFrame) -> None:
    """
    Check for FVG mitigation based on price action

    Pine Script logic:
    - Bull FVG mitigated when: close < get.min
    - Bear FVG mitigated when: close > get.max
    """
    df_reset = df.reset_index(drop=True)
    original_index = df.index.copy()

    for fvg in fvg_list:
        if fvg.mitigated:
            continue

        # Check mitigation starting from the bar after FVG creation
        start_check = fvg.start_index + 1

        for i in range(start_check, len(df_reset)):
            current_close = df_reset.loc[i, 'close']

            # Check mitigation conditions
            if fvg.is_bullish and current_close < fvg.bottom:
                # Bullish FVG mitigated
                fvg.mitigated = True
                fvg.mitigation_time = original_index[i]
                fvg.mitigation_index = i
                fvg.mitigation_price = current_close
                break

            elif not fvg.is_bullish and current_close > fvg.top:
                # Bearish FVG mitigated
                fvg.mitigated = True
                fvg.mitigation_time = original_index[i]
                fvg.mitigation_index = i
                fvg.mitigation_price = current_close
                break


def get_fvg_statistics(fvg_list: List[FairValueGap]) -> Dict:
    """
    Calculate FVG statistics similar to the Pine Script dashboard
    """
    if not fvg_list:
        return {
            'total_bull_fvgs': 0,
            'total_bear_fvgs': 0,
            'bull_mitigated': 0,
            'bear_mitigated': 0,
            'bull_mitigation_rate': 0.0,
            'bear_mitigation_rate': 0.0
        }

    bull_fvgs = [fvg for fvg in fvg_list if fvg.is_bullish]
    bear_fvgs = [fvg for fvg in fvg_list if not fvg.is_bullish]

    bull_mitigated = sum(1 for fvg in bull_fvgs if fvg.mitigated)
    bear_mitigated = sum(1 for fvg in bear_fvgs if fvg.mitigated)

    bull_mitigation_rate = (
        bull_mitigated / len(bull_fvgs)) * 100 if bull_fvgs else 0
    bear_mitigation_rate = (
        bear_mitigated / len(bear_fvgs)) * 100 if bear_fvgs else 0

    return {
        'total_bull_fvgs': len(bull_fvgs),
        'total_bear_fvgs': len(bear_fvgs),
        'bull_mitigated': bull_mitigated,
        'bear_mitigated': bear_mitigated,
        'bull_mitigation_rate': bull_mitigation_rate,
        'bear_mitigation_rate': bear_mitigation_rate
    }


def get_unmitigated_fvgs(fvg_list: List[FairValueGap], unmitigated_levels: int = 0) -> List[FairValueGap]:
    """
    Get unmitigated FVGs

    Parameters:
    - unmitigated_levels: Number of levels to show (0 = show all, like TradingView default)
    """
    unmitigated = [fvg for fvg in fvg_list if not fvg.mitigated]
    # Sort by start time (most recent first)
    unmitigated.sort(key=lambda x: x.start_time, reverse=True)

    if unmitigated_levels > 0:
        return unmitigated[:unmitigated_levels]
    else:
        return unmitigated  # Show all when 0


def calculate_dynamic_fvg_levels(df: pd.DataFrame, fvg_list: List[FairValueGap]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate dynamic FVG levels similar to Pine Script dynamic mode

    In dynamic mode, the FVG levels adjust based on current price action
    """
    if not fvg_list:
        return None, None

    current_close = df['close'].iloc[-1]

    # Find the most recent bullish and bearish FVGs
    recent_bull_fvg = None
    recent_bear_fvg = None

    for fvg in reversed(fvg_list):
        if fvg.is_bullish and not fvg.mitigated and recent_bull_fvg is None:
            recent_bull_fvg = fvg
        elif not fvg.is_bullish and not fvg.mitigated and recent_bear_fvg is None:
            recent_bear_fvg = fvg

        if recent_bull_fvg and recent_bear_fvg:
            break

    # Calculate dynamic levels
    max_bull_level = None
    min_bull_level = None
    max_bear_level = None
    min_bear_level = None

    if recent_bull_fvg:
        # Dynamic bullish FVG: max_bull_fvg := math.max(math.min(close, max_bull_fvg), min_bull_fvg)
        max_bull_level = max(
            min(current_close, recent_bull_fvg.top), recent_bull_fvg.bottom)
        min_bull_level = recent_bull_fvg.bottom

    if recent_bear_fvg:
        # Dynamic bearish FVG: min_bear_fvg := math.min(math.max(close, min_bear_fvg), max_bear_fvg)
        min_bear_level = min(
            max(current_close, recent_bear_fvg.bottom), recent_bear_fvg.top)
        max_bear_level = recent_bear_fvg.top

    return (max_bull_level, min_bull_level), (max_bear_level, min_bear_level)


def plot_fair_value_gaps(df: pd.DataFrame, fvg_list: List[FairValueGap],
                         unmitigated_levels: int = 0,
                         show_mitigation_levels: bool = False,
                         extend_bars: int = 20,
                         dynamic_mode: bool = False,
                         bullish_color: str = '#089981',
                         bearish_color: str = '#f23645',
                         show_dashboard: bool = False):
    """
    Plot Fair Value Gaps matching TradingView settings

    Parameters match TradingView inputs:
    - unmitigated_levels: 0 = show all (TradingView default)
    - show_mitigation_levels: False (TradingView default)
    - extend_bars: 20 (TradingView default)
    - dynamic_mode: False (TradingView default)
    """
    plt.figure(figsize=(15, 10))

    # Plot price data
    plt.plot(df.index, df['close'], label='Close Price',
             color='#2c3e50', linewidth=1.5)

    # Filter FVGs to display
    if unmitigated_levels > 0:
        display_fvgs = get_unmitigated_fvgs(fvg_list, unmitigated_levels)
    else:
        display_fvgs = fvg_list  # Show all when 0

    # Plot static FVG boxes (default mode)
    if not dynamic_mode:
        for fvg in display_fvgs:
            # Skip mitigated FVGs if we're only showing unmitigated
            if unmitigated_levels > 0 and fvg.mitigated:
                continue

            # Calculate end time with extension
            if fvg.mitigated:
                end_time = fvg.mitigation_time
            else:
                # Extend by specified number of bars
                total_bars = len(df)
                bars_from_start = total_bars - 1 - fvg.start_index
                if bars_from_start < extend_bars:
                    # Extend beyond current data
                    time_diff = df.index[-1] - \
                        df.index[-2] if len(df) > 1 else timedelta(minutes=15)
                    end_time = df.index[-1] + time_diff * \
                        (extend_bars - bars_from_start)
                else:
                    end_time = df.index[-1]

            # Color and transparency based on type and status
            if fvg.is_bullish:
                color = bullish_color
                alpha = 0.3 if not fvg.mitigated else 0.15
            else:
                color = bearish_color
                alpha = 0.3 if not fvg.mitigated else 0.15

            # Plot FVG rectangle
            plt.fill_betweenx([fvg.bottom, fvg.top], fvg.start_time, end_time,
                              color=color, alpha=alpha, edgecolor=color, linewidth=0.5)

            # Plot mitigation line if enabled and mitigated
            if show_mitigation_levels and fvg.mitigated:
                mitigation_level = fvg.bottom if fvg.is_bullish else fvg.top
                plt.plot([fvg.start_time, fvg.mitigation_time],
                         [mitigation_level, mitigation_level],
                         color=color, linestyle='--', linewidth=1, alpha=0.8)

    # Plot dynamic FVG levels
    if dynamic_mode:
        bull_levels, bear_levels = calculate_dynamic_fvg_levels(df, fvg_list)

        if bull_levels and bull_levels[0] is not None:
            max_bull, min_bull = bull_levels
            plt.fill_between(df.index, min_bull, max_bull,
                             color=bullish_color, alpha=0.3, label='Dynamic Bull FVG')

        if bear_levels and bear_levels[0] is not None:
            max_bear, min_bear = bear_levels
            plt.fill_between(df.index, min_bear, max_bear,
                             color=bearish_color, alpha=0.3, label='Dynamic Bear FVG')

    # Plot unmitigated level lines (when unmitigated_levels > 0)
    if unmitigated_levels > 0:
        unmitigated_fvgs = get_unmitigated_fvgs(fvg_list, unmitigated_levels)
        for fvg in unmitigated_fvgs:
            # Draw horizontal lines at FVG levels
            plt.axhline(y=fvg.top, color=bullish_color if fvg.is_bullish else bearish_color,
                        linestyle='-', linewidth=1.5, alpha=0.8)
            plt.axhline(y=fvg.bottom, color=bullish_color if fvg.is_bullish else bearish_color,
                        linestyle='-', linewidth=1.5, alpha=0.8)

    # Add dashboard if enabled
    if show_dashboard:
        stats = get_fvg_statistics(fvg_list)
        dashboard_text = (
            f"Bullish: {stats['total_bull_fvgs']} ({stats['bull_mitigation_rate']:.0f}% mitigated)\n"
            f"Bearish: {stats['total_bear_fvgs']} ({stats['bear_mitigation_rate']:.0f}% mitigated)"
        )
        plt.text(0.02, 0.98, dashboard_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Format chart
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.title('Fair Value Gap Detection (LuxAlgo Algorithm)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def analyze_fvg_trading_opportunities(df: pd.DataFrame, fvg_list: List[FairValueGap]) -> List[Dict]:
    """
    Analyze current trading opportunities based on unmitigated FVGs
    """
    if not fvg_list:
        return []

    current_price = df['close'].iloc[-1]
    current_time = df.index[-1]

    opportunities = []
    unmitigated_fvgs = get_unmitigated_fvgs(fvg_list, 0)  # Get all unmitigated

    for fvg in unmitigated_fvgs:
        # Calculate distance from current price
        if fvg.is_bullish:
            # For bullish FVG, we're interested if price is above the gap (potential support)
            if current_price > fvg.top:
                distance_pct = ((current_price - fvg.top) / fvg.top) * 100
                opportunity_type = "Support Test"
                entry_level = fvg.top
                stop_level = fvg.bottom
            else:
                continue  # Price hasn't cleared the FVG yet
        else:
            # For bearish FVG, we're interested if price is below the gap (potential resistance)
            if current_price < fvg.bottom:
                distance_pct = ((fvg.bottom - current_price) /
                                current_price) * 100
                opportunity_type = "Resistance Test"
                entry_level = fvg.bottom
                stop_level = fvg.top
            else:
                continue  # Price hasn't cleared the FVG yet

        # Calculate age of FVG
        age_hours = (current_time - fvg.start_time).total_seconds() / 3600

        # Calculate strength score based on multiple factors
        strength_score = 50  # Base score

        # Age factor (newer FVGs are generally more relevant)
        if age_hours <= 24:
            strength_score += 20
        elif age_hours <= 72:
            strength_score += 10
        elif age_hours > 168:  # Older than 1 week
            strength_score -= 20

        # Size factor (larger gaps are more significant)
        gap_height = fvg.get_height()
        avg_range = df['high'].subtract(df['low']).rolling(20).mean().iloc[-1]
        size_ratio = gap_height / avg_range if avg_range > 0 else 1

        if size_ratio > 1.5:
            strength_score += 15
        elif size_ratio > 1.0:
            strength_score += 10
        elif size_ratio < 0.5:
            strength_score -= 10

        # Distance factor (closer to current price is more actionable)
        if distance_pct <= 2:
            strength_score += 15
        elif distance_pct <= 5:
            strength_score += 10
        elif distance_pct > 10:
            strength_score -= 15

        opportunity = {
            'fvg': fvg,
            'type': opportunity_type,
            'direction': 'Bull' if fvg.is_bullish else 'Bear',
            'current_price': current_price,
            'entry_level': entry_level,
            'stop_level': stop_level,
            'gap_height': gap_height,
            'distance_pct': distance_pct,
            'age_hours': age_hours,
            'strength_score': max(0, min(100, strength_score)),
            'risk_reward_ratio': abs(entry_level - stop_level) / gap_height if gap_height > 0 else 0
        }

        opportunities.append(opportunity)

    # Sort by strength score
    opportunities.sort(key=lambda x: x['strength_score'], reverse=True)

    return opportunities


def detect_and_analyze_fvgs(df: pd.DataFrame,
                            threshold_pct: float = 0.0,
                            auto_threshold: bool = False,
                            unmitigated_levels: int = 0,
                            show_mitigation_levels: bool = False,
                            timeframe: str = "chart",
                            extend_bars: int = 20,
                            dynamic_mode: bool = False,
                            show_dashboard: bool = False) -> Tuple[List[FairValueGap], Dict, List[Dict]]:
    """
    Main function to detect and analyze Fair Value Gaps with TradingView-matching parameters

    Default values match TradingView settings:
    - threshold_pct: 0.0
    - auto_threshold: False
    - unmitigated_levels: 0 (show all)
    - show_mitigation_levels: False
    - extend_bars: 20
    - dynamic_mode: False
    - show_dashboard: False
    """
    # Detect FVGs
    fvg_list = detect_fair_value_gaps(
        df, threshold_pct, auto_threshold, timeframe)

    # Check for mitigation
    check_fvg_mitigation(fvg_list, df)

    # Get statistics
    stats = get_fvg_statistics(fvg_list) if show_dashboard else {}

    # Analyze opportunities
    opportunities = analyze_fvg_trading_opportunities(df, fvg_list)

    return fvg_list, stats, opportunities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fair Value Gap Detector (LuxAlgo Algorithm)')

    # Data parameters
    parser.add_argument('--symbol', type=str,
                        default='BTCUSDT', help='Symbol to fetch data')
    parser.add_argument('--interval', type=str, default='15m',
                        help='Interval to fetch data')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days of historical data')

    # TradingView matching parameters with exact defaults
    parser.add_argument('--threshold_pct', type=float, default=0.0,
                        help='Threshold percentage (default: 0.0, matches TradingView)')
    parser.add_argument('--auto_threshold', type=str, default='false',
                        choices=['true', 'false'], help='Use automatic threshold (default: false)')
    parser.add_argument('--unmitigated_levels', type=int, default=0,
                        help='Unmitigated levels to show (0 = show all, matches TradingView default)')
    parser.add_argument('--mitigation_levels', type=str, default='false',
                        choices=['true', 'false'], help='Show mitigation levels (default: false)')
    parser.add_argument('--timeframe', type=str, default='chart',
                        help='Timeframe for detection (default: chart)')
    parser.add_argument('--extend', type=int, default=20,
                        help='Extend bars (default: 20, matches TradingView)')
    parser.add_argument('--dynamic', type=str, default='false',
                        choices=['true', 'false'], help='Dynamic mode (default: false)')
    parser.add_argument('--dashboard', type=str, default='false',
                        choices=['true', 'false'], help='Show dashboard (default: false)')

    args = parser.parse_args()

    # Fetch data
    client = Client()
    start_time = datetime.now() - timedelta(days=args.days)

    data_fetcher = BinanceDataFetcher(client=client)
    df = data_fetcher.get_historical_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_time=start_time
    )

    if df.empty:
        print("❌ No data fetched")
        exit(1)

    # Convert string arguments to boolean
    auto_threshold = args.auto_threshold.lower() == 'true'
    show_mitigation_levels = args.mitigation_levels.lower() == 'true'
    dynamic_mode = args.dynamic.lower() == 'true'
    show_dashboard = args.dashboard.lower() == 'true'

    # Detect and analyze FVGs with TradingView-matching settings
    fvg_list, stats, opportunities = detect_and_analyze_fvgs(
        df,
        threshold_pct=args.threshold_pct,
        auto_threshold=auto_threshold,
        unmitigated_levels=args.unmitigated_levels,
        show_mitigation_levels=show_mitigation_levels,
        timeframe=args.timeframe,
        extend_bars=args.extend,
        dynamic_mode=dynamic_mode,
        show_dashboard=show_dashboard
    )

    # Display results
    print(f"\n=== FAIR VALUE GAP ANALYSIS (TradingView Settings) ===")
    print(f"Symbol: {args.symbol} | Timeframe: {args.interval}")
    print(
        f"Threshold: {'Auto' if auto_threshold else f'{args.threshold_pct}%'}")
    print(f"Unmitigated Levels: {args.unmitigated_levels} (0 = show all)")
    print(
        f"Mitigation Levels: {'Enabled' if show_mitigation_levels else 'Disabled'}")
    print(f"Extend: {args.extend} bars")
    print(f"Dynamic Mode: {'Enabled' if dynamic_mode else 'Disabled'}")
    print(f"Total FVGs Detected: {len(fvg_list)}")

    # Show statistics if dashboard enabled
    if show_dashboard and stats:
        print(f"\n--- 📊 DASHBOARD ---")
        print(
            f"Bullish FVGs: {stats['total_bull_fvgs']} (Mitigated: {stats['bull_mitigated']}, {stats['bull_mitigation_rate']:.1f}%)")
        print(
            f"Bearish FVGs: {stats['total_bear_fvgs']} (Mitigated: {stats['bear_mitigated']}, {stats['bear_mitigation_rate']:.1f}%)")

    # Show FVGs based on unmitigated_levels setting
    if args.unmitigated_levels > 0:
        display_fvgs = get_unmitigated_fvgs(fvg_list, args.unmitigated_levels)
        print(f"\n--- 🟢 UNMITIGATED FVGs (Showing {len(display_fvgs)}) ---")
    else:
        display_fvgs = fvg_list
        unmitigated_count = len([fvg for fvg in fvg_list if not fvg.mitigated])
        print(
            f"\n--- 📊 ALL FVGs (Total: {len(fvg_list)}, Unmitigated: {unmitigated_count}) ---")

    for i, fvg in enumerate(display_fvgs[:10]):  # Show max 10
        fvg_type = "BULL" if fvg.is_bullish else "BEAR"
        status = "MITIGATED" if fvg.mitigated else "ACTIVE"
        print(
            f"{i+1}. {fvg_type} [{status}] | {fvg.start_time} | {fvg.bottom:.4f} - {fvg.top:.4f} | Height: {fvg.get_height():.6f}")

    # Show trading opportunities
    if opportunities:
        print(f"\n--- 🎯 TRADING OPPORTUNITIES ---")
        for i, opp in enumerate(opportunities[:5]):  # Show top 5
            direction_emoji = "🟢" if opp['direction'] == 'Bull' else "🔴"
            print(
                f"{i+1}. {direction_emoji} {opp['direction']} {opp['type']} | Score: {opp['strength_score']:.0f}/100")
            print(
                f"   Entry: {opp['entry_level']:.4f} | Stop: {opp['stop_level']:.4f} | Distance: {opp['distance_pct']:.1f}% | Age: {opp['age_hours']:.1f}h")

    # Plot results with TradingView-matching settings
    plot_fair_value_gaps(
        df, fvg_list,
        unmitigated_levels=args.unmitigated_levels,
        show_mitigation_levels=show_mitigation_levels,
        extend_bars=args.extend,
        dynamic_mode=dynamic_mode,
        show_dashboard=show_dashboard
    )
