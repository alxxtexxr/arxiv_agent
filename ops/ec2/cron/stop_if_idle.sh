#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ec2-user/arxiv_agent"
ACTIVITY_FILE="$PROJECT_DIR/.last_activity"
IDLE_TIMEOUT=1800   # 30 minutes

if [[ ! -f "$ACTIVITY_FILE" ]]; then
    echo "No activity file yet – skipping."
    exit 0
fi

LAST_ACTIVITY=$(cat "$ACTIVITY_FILE")
NOW=$(date +%s)
ELAPSED=$((NOW - LAST_ACTIVITY))

if [[ $ELAPSED -gt $IDLE_TIMEOUT ]]; then
    echo "$(date): No activity for $ELAPSED seconds. Stopping instance..."
    # Load environment variables (e.g., INSTANCE_CONTROL_API_KEY)
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        set -a; source "$PROJECT_DIR/.env"; set +a
    fi
    API_URL="https://instance-control-api.alimtegar404.workers.dev/v1/instances/arxiv-agent/stop"
    API_KEY="${INSTANCE_CONTROL_API_KEY:-}"
    if [[ -n "$API_KEY" ]]; then
        curl -X POST "$API_URL" -H "X-Api-Key: $API_KEY" --fail --max-time 30
    else
        # Fallback to AWS CLI (ensure IAM role with ec2:StopInstances)
        INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
    fi
else
    echo "$(date): Activity $ELAPSED sec ago – no action"
fi