#!/usr/bin/env bash
# Install all systemd services on the EC2 instance.
# Run this once via SSH: bash ops/ec2/systemd/install.sh

set -euo pipefail

SERVICE_DIR="/etc/systemd/system"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Copying service files..."
sudo cp "$SCRIPT_DIR/pgvector.service" "$SERVICE_DIR/"
sudo cp "$SCRIPT_DIR/arxiv-agent.service" "$SERVICE_DIR/"
sudo cp "$SCRIPT_DIR/arxiv-agent-daily-agent.service" "$SERVICE_DIR/"

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling services..."
sudo systemctl enable pgvector.service
sudo systemctl enable arxiv-agent.service
sudo systemctl enable arxiv-agent-daily-agent.service

echo "Done. Services will start on boot:"
echo "  1. pgvector (vector database)"
echo "  2. arxiv-agent (LangGraph agent server)"
echo "  3. arxiv-agent-daily-agent (embedding + bookmarks, then auto-stop)"
echo ""
echo "Check status: systemctl status pgvector arxiv-agent arxiv-agent-daily-agent"
