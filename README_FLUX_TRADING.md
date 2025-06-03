# 🚀 Flux Order Block Trading Bot

Advanced live trading bot sử dụng thuật toán **Flux Chart Order Block** với hệ thống **Entry Evaluation** đa yếu tố. Bot tập trung vào **High-Quality Setups**, **Precise Entry**, và **Stop Loss Management**.

## ✨ Tính năng nổi bật

### 🔍 Flux Order Block Detection
- **Pine Script Accuracy**: Thuật toán detect order blocks chính xác như TradingView
- **Swing Point Detection**: Sử dụng phương pháp `ta.highest/lowest` của Pine Script
- **Smart Filtering**: Lọc theo ATR, distance, volume và overlap percentage
- **Multi-timeframe Support**: Hỗ trợ các timeframe khác nhau (5m, 15m, 1h, 4h)

### 🎯 Advanced Entry Evaluation System
- **4-Factor Scoring System**:
  - **Trend Analysis (40%)**: Multi-timeframe trend confluence
  - **Market Structure (25%)**: Higher highs/lows analysis
  - **Breaker Analysis (20%)**: Recently broken order blocks context
  - **Volume & Momentum (15%)**: Volume strength vs average
- **Adaptive Thresholds**: Dynamic entry requirements dựa trên market conditions
- **Smart Warnings**: Cảnh báo counter-trend setups và risks

### 📊 Multi-timeframe Trend Analysis
- **Primary Trend Detection**: EMA 21/50/100 analysis
- **Supporting Timeframes**: 3x, 6x, 12x higher timeframe confirmation
- **Trend Change Detection**: Identifies potential reversal points
- **Confluence Scoring**: 0-100% confluence rating

### ⚠️ Intelligent Risk Management
- **Position Sizing**: Dựa trên risk per trade và capital allocation
- **Dynamic Stop Loss**: Calculated từ order block structure
- **Take Profit Optimization**: Risk/reward optimization
- **Distance Filtering**: Limit orders theo khoảng cách từ current price

## 🛠️ Cài đặt & Setup

### 1. Requirements
```bash
pip install pandas numpy requests python-binance python-dotenv matplotlib
```

### 2. Configuration
```bash
# Copy config template
cp flux_trading_config.example .env

# Edit .env với API keys và settings
```

### 3. Quick Start
```bash
# Setup
python start_flux_trading.py --setup

# Test algorithm
python start_flux_trading.py --test-algo

# Start in test mode
python start_flux_trading.py --start --test

# Start live trading (careful!)
python start_flux_trading.py --start --live
```

## ⚙️ Configuration chi tiết

### Trading Mode & Capital
```bash
TEST_MODE=true                    # Test mode (khuyến nghị)
TEST_CAPITAL=1000                 # Capital for test mode
MAX_CAPITAL=1000                  # Max capital for live mode
```

### Market Configuration
```bash
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT,SOLUSDT
TRADING_INTERVAL=15m              # 5m, 15m, 1h, 4h
LOOKBACK_HOURS=72                 # Historical data lookback
```

### Flux Order Block Parameters
```bash
SWING_LENGTH=10                   # Swing detection period
MAX_ATR_MULT=3.5                  # Max OB size filter
MITIGATION_METHOD=Wick            # Wick or Close
MAX_BULLISH_OBS=5                 # Max bullish OBs to track
MAX_BEARISH_OBS=5                 # Max bearish OBs to track
```

### Entry Evaluation System
```bash
USE_ENTRY_EVALUATION=true         # Enable advanced filtering
ENTRY_THRESHOLD=45                # Minimum score (0-100)
```

**Threshold Guidelines:**
- **45-55**: Moderate quality (more signals, higher risk)
- **55-70**: Good quality (balanced approach) ⭐ **Recommended**
- **70-85**: High quality (fewer but better signals)
- **85+**: Exceptional quality (very selective)

### Risk Management
```bash
RISK_PER_TRADE_PCT=2.0            # Risk per trade (1-3% recommended)
LEVERAGE=10                       # Leverage (5-20x)
STOP_LOSS_PCT=2.0                 # Stop loss percentage
TAKE_PROFIT_PCT=4.0               # Take profit percentage (2:1 RR)
MAX_DISTANCE_PCT=5.0              # Max distance from current price
```

## 📱 Telegram Notifications

Bot gửi thông báo chi tiết về:

### 🟢/🔴 Order Block Signals
```
🟢 FLUX ORDER BLOCK LONG

Symbol: BTCUSDT
Entry: $43,250.0000
Stop Loss: $42,385.0000
Take Profit: $44,980.0000
Position Size: $40.00
Risk/Reward: 2.00
━━━━━━━━━━━━━━━━━━━━━━
📊 Entry Analysis
Score: 78.5/100
Quality: Good
Risk Level: Low-Medium
Trend: Bullish
Supporting TFs: 2/3
━━━━━━━━━━━━━━━━━━━━━━
📊 Order Block Details
Range: 43,180.0000 - 43,320.0000
Height: 0.001400
Volume: 28574
Status: 🟢 Active
Created: 12-25 14:30
```

### ✅ Order Fills & Position Management
- Real-time fill notifications
- Stop loss và take profit confirmations
- PnL tracking với percentage gains/losses

## 🔧 Advanced Usage

### Command Line Options
```bash
# Override symbols
python start_flux_trading.py --start --symbols BTCUSDT,ETHUSDT

# Override entry threshold
python start_flux_trading.py --start --threshold 60

# Force test mode
python start_flux_trading.py --start --test

# Show current config
python start_flux_trading.py --config
```

### Algorithm Testing
```bash
# Test flux algorithm với live data
python start_flux_trading.py --test-algo
```

## 📊 Entry Evaluation Factors

### 1. Trend Analysis (40% weight)
- **Primary Trend**: EMA 21/50/100 direction
- **Supporting Timeframes**: Higher TF confirmation
- **Trend Strength**: Momentum and consistency
- **Reversal Detection**: Potential trend changes

### 2. Market Structure (25% weight)
- **Higher Highs/Lows**: Structure pattern analysis
- **Momentum Shifts**: Recent price action
- **Support/Resistance**: Key level interactions

### 3. Breaker Analysis (20% weight)
- **Recent Breakers**: Recently broken order blocks
- **Confluence/Divergence**: Supporting vs opposing breakers
- **Breaker Strength**: Volume of breaking candles

### 4. Volume & Momentum (15% weight)
- **Relative Volume**: Current vs average volume
- **Volume Confirmation**: Order block formation volume
- **Momentum Indicators**: RSI and velocity analysis

## 🚦 Safety Features

### Test Mode
- **Complete Simulation**: Full order flow simulation
- **No Real Money**: Zero financial risk
- **Real Market Data**: Accurate price simulation
- **Full Logging**: Complete activity tracking

### Risk Controls
- **Position Limits**: Max orders per symbol
- **Capital Protection**: Maximum risk per trade
- **Distance Limits**: Prevent far-away entries
- **Age Limits**: Auto-cancel old orders

### Error Handling
- **Network Resilience**: Auto-retry on connection issues
- **API Rate Limits**: Intelligent request spacing
- **Data Validation**: Input validation and sanitization
- **Graceful Shutdown**: Clean exit with order cancellation

## 📈 Performance Metrics

### Entry Quality Distribution
| Score Range | Quality | Typical Win Rate | Risk Level |
|-------------|---------|------------------|------------|
| 85-100 | Excellent | 70-80% | Very Low |
| 70-84 | Good | 60-70% | Low |
| 55-69 | Moderate | 50-60% | Medium |
| 45-54 | Fair | 40-50% | High |
| <45 | Poor | <40% | Very High |

### Risk-Reward Optimization
- **Default RR**: 2:1 (4% TP, 2% SL)
- **Adaptive Sizing**: Position size based on setup quality
- **Max Risk**: Configurable per-trade risk limit

## 🔍 Monitoring & Logging

### Log Files
- **flux_trading.log**: Detailed bot activity
- **Console Output**: Real-time status updates
- **Telegram**: Live notifications

### Status Information
```bash
# Every 5 minutes status update
Status: 3 orders, 1 positions
BTCUSDT: Analyzing for Flux Order Blocks...
ETHUSDT: Generated 1 orders from 2 active OBs
```

## 🚨 Troubleshooting

### Common Issues

**"No quality order blocks found"**
- Lower entry threshold (45-55)
- Check market conditions (trending vs sideways)
- Verify sufficient historical data

**"Algorithm test failed"**
- Check API keys và network connection
- Verify symbol availability
- Check dependencies installation

**"Invalid quantity"**
- Check symbol precision settings
- Verify capital allocation
- Review position sizing calculation

**"Distance too far"**
- Increase MAX_DISTANCE_PCT
- Wait for price to approach order blocks
- Consider different timeframes

## 📞 Support & Optimization

### Performance Tuning
1. **Conservative**: Threshold 65+, Risk 1%
2. **Balanced**: Threshold 50-65, Risk 2% ⭐
3. **Aggressive**: Threshold 45-55, Risk 3%

### Best Practices
- Always start với TEST_MODE=true
- Monitor Telegram notifications closely
- Regularly review log files
- Adjust threshold based on market conditions
- Use multiple symbols for diversification

### Market Conditions
- **Trending Markets**: Lower threshold (45-55)
- **Sideways Markets**: Higher threshold (60-70)
- **Volatile Markets**: Reduce leverage và position size
- **Low Volume**: Increase distance và be more selective

---

**🎯 Flux Order Block Trading Bot - Where Precision Meets Profitability 🎯** 