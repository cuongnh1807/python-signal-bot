#!/bin/bash

# Kill any existing instances
pkill -f "python3 cron_trading.py"

# Start the bot in background
nohup python3 cron_trading.py > trading_bot.log 2>&1 &

# Get the PID of the new process
PID=$!
echo "Trading bot started with PID: $PID"
echo "PID $PID" > bot_pid.txt
echo "Check trading_bot.log for output"