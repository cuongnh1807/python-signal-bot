#!/bin/bash

if [ -f bot_pid.txt ]; then
    PID=$(cat bot_pid.txt | cut -d' ' -f2)
    kill $PID
    rm bot_pid.txt
    echo "Trading bot stopped (PID: $PID)"
else
    echo "No running bot found"
fi