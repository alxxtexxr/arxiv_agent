#!/usr/bin/env bash
# Install cron jobs on the EC2 instance.
# Run this once via SSH: bash ops/ec2/cron/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="/usr/local/bin/arxiv-agent-cron"

echo "Installing cron scripts to $INSTALL_DIR..."
sudo mkdir -p "$INSTALL_DIR"
sudo cp "$SCRIPT_DIR/stop_if_idle.sh" "$INSTALL_DIR/"
sudo chmod +x "$INSTALL_DIR/stop_if_idle.sh"

echo "Installing cron entry..."
# Run stop_if_idle.sh every 5 minutes
CRON_LINE="*/5 * * * * $INSTALL_DIR/stop_if_idle.sh >> /home/ec2-user/auto-stop.log 2>&1"
# Add the cron line if it doesn't already exist
if ! crontab -l 2>/dev/null | grep -qF "arxiv-agent-cron/stop_if_idle.sh"; then
    (
        crontab -l 2>/dev/null || true
        echo "$CRON_LINE"
    ) | crontab -
fi

echo "Done. Cron job installed:"
echo "  stop_if_idle.sh  →  every 5 minutes (checks idle timeout, stops instance if inactive)"
echo ""
echo "Verify: crontab -l"
