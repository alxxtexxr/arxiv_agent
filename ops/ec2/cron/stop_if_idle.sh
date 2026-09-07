#!/usr/bin/env bash

# Configuration
LOG_FILE="/home/ec2-user/arxiv_agent/arxiv-agent.log"
PORT=2024
IDLE_MINUTES=30
CHECK_INTERVAL_MINUTES=5

# Use the log file's last modification time as a proxy for recent activity
if [ -f "$LOG_FILE" ]; then
    # Convert to epoch seconds (Linux)
    LAST_MOD=$(stat -c %Y "$LOG_FILE" 2>/dev/null || stat -f %m "$LOG_FILE" 2>/dev/null)
    NOW=$(date +%s)
    AGE=$(( (NOW - LAST_MOD) / 60 ))   # age in minutes
else
    AGE=9999   # if log file doesn't exist, treat as idle
fi

# Additionally, check if there are any active connections to the agent port.
# If there are, we definitely do not want to stop.
ACTIVE_CONNS=$(ss -tn state established "( sport = :$PORT or dport = :$PORT )" | grep -c -v "^State" || true)

# Stop only if:
#   - log file hasn't been touched in IDLE_MINUTES or more
#   - AND there are no active connections (to avoid interrupting a running request)
if [ $AGE -ge $IDLE_MINUTES ] && [ $ACTIVE_CONNS -eq 0 ]; then
    echo "$(date): No activity for $IDLE_MINUTES minutes, stopping instance." >> /home/ec2-user/auto-stop.log
    # Stop the instance gracefully
    sudo shutdown -h now
fi