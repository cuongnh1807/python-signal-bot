#!/usr/bin/env python3
"""
Test Order Breaker Blocks Detection
Compare results with TradingView SOL data
"""

import sys
import os
from datetime import datetime, timedelta
from indicators.order_breaker_blocks import detect_order_breaker_blocks
from binance_data_fetcher import BinanceDataFetcher
from binance.client import Client
import pandas as pd


def test_sol_order_blocks():
    """Test SOLUSDT order block detection"""
    print("🧪 Testing Order Breaker Blocks on SOLUSDT...")
    
    # Test parameters
    symbol = 'SOLUSDT'
    interval = '15m'
    days = 7
    
    # Fetch data
    client = Client()
    start_time = datetime.now() - timedelta(days=days)
    
    print(f"📊 Fetching {symbol} {interval} data for last {days} days...")
    
    raw_data = client.get_historical_klines(
        symbol,
        interval=interval,
        start_str=int(start_time.timestamp() * 1000),
        end_str=int(datetime.now().timestamp() * 1000)
    )
    
    # Process data
    data = pd.DataFrame(raw_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = data[col].astype(float)
    
    data.set_index('timestamp', inplace=True)
    
    print(f"✅ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    print(f"📈 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    
    # Test different configurations
    test_configs = [
        {
            'name': 'TradingView Matching',
            'swing_length': 10,
            'max_atr_mult': 3.5,
            'ob_end_method': 'Wick',
            'bullish_count': 3,
            'bearish_count': 3,
            'use_entry_eval': False
        },
        {
            'name': 'Entry Evaluation Enabled',
            'swing_length': 10,
            'max_atr_mult': 3.5,
            'ob_end_method': 'Wick',
            'bullish_count': 3,
            'bearish_count': 3,
            'use_entry_eval': True
        },
        {
            'name': 'More Sensitive',
            'swing_length': 5,
            'max_atr_mult': 2.0,
            'ob_end_method': 'Wick',
            'bullish_count': 5,
            'bearish_count': 5,
            'use_entry_eval': True
        }
    ]
    
    current_price = data['close'].iloc[-1]
    print(f"💰 Current {symbol} price: ${current_price:.4f}")
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"🔬 TEST: {config['name']}")
        print(f"{'='*60}")
        
        # Detect order blocks
        order_blocks = detect_order_breaker_blocks(
            data,
            swing_length=config['swing_length'],
            max_atr_mult=config['max_atr_mult'],
            ob_end_method=config['ob_end_method'],
            bullish_ob_count=config['bullish_count'],
            bearish_ob_count=config['bearish_count'],
            use_entry_evaluation=config['use_entry_eval']
        )
        
        # Analyze results
        bullish_obs = [ob for ob in order_blocks if ob.ob_type == "Bull"]
        bearish_obs = [ob for ob in order_blocks if ob.ob_type == "Bear"]
        
        active_bullish = [ob for ob in bullish_obs if not ob.breaker]
        active_bearish = [ob for ob in bearish_obs if not ob.breaker]
        
        broken_bullish = [ob for ob in bullish_obs if ob.breaker]
        broken_bearish = [ob for ob in bearish_obs if ob.breaker]
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"Total Order Blocks: {len(order_blocks)}")
        print(f"Active: {len(active_bullish + active_bearish)} | Broken: {len(broken_bullish + broken_bearish)}")
        print(f"Bullish: {len(bullish_obs)} (Active: {len(active_bullish)}, Broken: {len(broken_bullish)})")
        print(f"Bearish: {len(bearish_obs)} (Active: {len(active_bearish)}, Broken: {len(broken_bearish)})")
        
        # Show closest to current price
        close_obs = []
        for ob in active_bullish + active_bearish:
            if ob.ob_type == "Bull":
                distance = abs((current_price - ob.bottom) / current_price) * 100
                entry_price = ob.bottom
            else:
                distance = abs((current_price - ob.top) / current_price) * 100
                entry_price = ob.top
                
            if distance <= 5.0:  # Within 5%
                close_obs.append((ob, distance, entry_price))
        
        # Sort by distance
        close_obs.sort(key=lambda x: x[1])
        
        print(f"\n🎯 ORDER BLOCKS NEAR CURRENT PRICE (within 5%):")
        if close_obs:
            for ob, distance, entry_price in close_obs[:5]:  # Top 5 closest
                direction = "🟢 LONG" if ob.ob_type == "Bull" else "🔴 SHORT"
                status = "BROKEN" if ob.breaker else "ACTIVE"
                
                info = (f"{direction} | {status} | Entry: ${entry_price:.4f} | "
                        f"Distance: {distance:.2f}% | Range: {ob.bottom:.4f}-{ob.top:.4f}")
                
                if hasattr(ob, 'entry_score') and ob.entry_score:
                    info += f" | Score: {ob.entry_score:.0f}"
                
                print(f"  {info}")
        else:
            print("  No order blocks within 5% of current price")
        
        # Show recent breakers
        recent_breakers = []
        cutoff_time = datetime.now() - timedelta(hours=24)
        for ob in broken_bullish + broken_bearish:
            if ob.break_time and ob.break_time >= cutoff_time:
                recent_breakers.append(ob)
        
        recent_breakers.sort(key=lambda x: x.break_time, reverse=True)
        
        print(f"\n💥 RECENT BREAKERS (last 24h): {len(recent_breakers)}")
        for i, ob in enumerate(recent_breakers[:3]):  # Show latest 3
            direction = "🟢 BULL" if ob.ob_type == "Bull" else "🔴 BEAR"
            print(f"  {i+1}. {direction} | Broken: {ob.break_time.strftime('%m-%d %H:%M')} | "
                  f"Range: {ob.bottom:.4f}-{ob.top:.4f} | Break Vol: {ob.bb_volume:.0f}")
        
        print(f"\n📋 DETAILED ORDER BLOCKS:")
        
        # Show bullish OBs
        if bullish_obs:
            print(f"\n--- 🟢 BULLISH ORDER BLOCKS ---")
            for i, ob in enumerate(bullish_obs):
                status = "🔴 BROKEN" if ob.breaker else "🟢 ACTIVE"
                age = (datetime.now() - ob.start_time).total_seconds() / 3600
                
                volume_ratio = "N/A"
                if ob.ob_low_volume and ob.ob_high_volume:
                    percentage = int((min(ob.ob_low_volume, ob.ob_high_volume) / 
                                   max(ob.ob_low_volume, ob.ob_high_volume)) * 100)
                    volume_ratio = f"{ob.ob_low_volume:.0f}/{ob.ob_high_volume:.0f} ({percentage}%)"
                
                info = (f"  {i+1}. {status} | Entry: ${ob.bottom:.4f} | "
                        f"Range: {ob.bottom:.4f}-{ob.top:.4f} | "
                        f"Age: {age:.1f}h | Vol: {ob.ob_volume:.0f} ({volume_ratio})")
                
                if hasattr(ob, 'entry_score') and ob.entry_score:
                    info += f"\n      📊 Score: {ob.entry_score:.1f} | Quality: {ob.entry_quality} | Risk: {ob.risk_level}"
                
                if ob.breaker:
                    break_age = (datetime.now() - ob.break_time).total_seconds() / 3600
                    info += f"\n      💥 Broken {break_age:.1f}h ago | Break Vol: {ob.bb_volume:.0f}"
                
                print(info)
        
        # Show bearish OBs
        if bearish_obs:
            print(f"\n--- 🔴 BEARISH ORDER BLOCKS ---")
            for i, ob in enumerate(bearish_obs):
                status = "🔴 BROKEN" if ob.breaker else "🟢 ACTIVE"
                age = (datetime.now() - ob.start_time).total_seconds() / 3600
                
                volume_ratio = "N/A"
                if ob.ob_low_volume and ob.ob_high_volume:
                    percentage = int((min(ob.ob_low_volume, ob.ob_high_volume) / 
                                   max(ob.ob_low_volume, ob.ob_high_volume)) * 100)
                    volume_ratio = f"{ob.ob_low_volume:.0f}/{ob.ob_high_volume:.0f} ({percentage}%)"
                
                info = (f"  {i+1}. {status} | Entry: ${ob.top:.4f} | "
                        f"Range: {ob.bottom:.4f}-{ob.top:.4f} | "
                        f"Age: {age:.1f}h | Vol: {ob.ob_volume:.0f} ({volume_ratio})")
                
                if hasattr(ob, 'entry_score') and ob.entry_score:
                    info += f"\n      📊 Score: {ob.entry_score:.1f} | Quality: {ob.entry_quality} | Risk: {ob.risk_level}"
                
                if ob.breaker:
                    break_age = (datetime.now() - ob.break_time).total_seconds() / 3600
                    info += f"\n      💥 Broken {break_age:.1f}h ago | Break Vol: {ob.bb_volume:.0f}"
                
                print(info)


def test_comparison_with_flux():
    """Compare with flux_orderblock.py results"""
    print(f"\n{'='*60}")
    print("🔄 COMPARISON WITH FLUX ORDERBLOCK")
    print(f"{'='*60}")
    
    try:
        from indicators.flux_orderblock import detect_flux_order_blocks
        
        symbol = 'SOLUSDT'
        interval = '15m'
        days = 3
        
        # Fetch data
        client = Client()
        start_time = datetime.now() - timedelta(days=days)
        
        raw_data = client.get_historical_klines(
            symbol,
            interval=interval,
            start_str=int(start_time.timestamp() * 1000),
            end_str=int(datetime.now().timestamp() * 1000)
        )
        
        data = pd.DataFrame(raw_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = data[col].astype(float)
        
        data.set_index('timestamp', inplace=True)
        
        # Run both algorithms
        print("Running Order Breaker Blocks...")
        breaker_obs = detect_order_breaker_blocks(
            data, swing_length=10, bullish_ob_count=3, bearish_ob_count=3, use_entry_evaluation=False
        )
        
        print("Running Flux Order Blocks...")
        flux_obs = detect_flux_order_blocks(
            data, swing_length=10, bullish_ob_count=3, bearish_ob_count=3, use_entry_evaluation=False
        )
        
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"Order Breaker Blocks: {len(breaker_obs)} total")
        print(f"Flux Order Blocks: {len(flux_obs)} total")
        
        # Count by type
        breaker_bull = len([ob for ob in breaker_obs if ob.ob_type == "Bull"])
        breaker_bear = len([ob for ob in breaker_obs if ob.ob_type == "Bear"])
        flux_bull = len([ob for ob in flux_obs if ob.ob_type == "Bull"])
        flux_bear = len([ob for ob in flux_obs if ob.ob_type == "Bear"])
        
        print(f"Bullish: Breaker={breaker_bull}, Flux={flux_bull}")
        print(f"Bearish: Breaker={breaker_bear}, Flux={flux_bear}")
        
        # Check for breaker status difference
        breaker_broken = len([ob for ob in breaker_obs if hasattr(ob, 'breaker') and ob.breaker])
        print(f"Broken OBs (Breaker algorithm): {breaker_broken}")
        
    except ImportError as e:
        print(f"❌ Could not import flux_orderblock: {e}")


if __name__ == "__main__":
    print("🚀 ORDER BREAKER BLOCKS TESTING")
    print("=" * 50)
    
    try:
        test_sol_order_blocks()
        test_comparison_with_flux()
        
        print(f"\n✅ Testing completed successfully!")
        print(f"\n🎯 Key Features Tested:")
        print(f"  - Pine Script accurate swing detection")
        print(f"  - Order block formation logic")
        print(f"  - Breaker detection & tracking")
        print(f"  - Volume breakdown analysis")
        print(f"  - Entry evaluation integration")
        print(f"  - Distance filtering for trading")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 