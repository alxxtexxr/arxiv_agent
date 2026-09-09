#!/usr/bin/env bash
# Install all systemd services on the EC2 instance.
# Run this once via SSH: bash ops/ec2/systemd/install.sh

set -euo pipefail

SERVICE_DIR="/etc/systemd/system"
ENV_FILE="/home/ec2-user/arxiv_agent/.env"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Hugging Face token (optional but recommended) ──────────────────────
echo ""
echo "Hugging Face token (optional)"
echo "  fastembed downloads models from huggingface.co."
echo "  An HF_TOKEN gives higher rate limits and avoids warnings."
echo "  Get one free at https://huggingface.co/settings/tokens"
echo ""
read -rp "Paste HF_TOKEN (leave blank to skip): " HF_TOKEN

if [[ -n "$HF_TOKEN" ]]; then
  # Append or replace HF_TOKEN in the remote .env
  if grep -q '^HF_TOKEN=' "$ENV_FILE" 2>/dev/null; then
    sudo sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" "$ENV_FILE"
  else
    echo "HF_TOKEN=$HF_TOKEN" | sudo tee -a "$ENV_FILE" >/dev/null
  fi
  echo "✓ HF_TOKEN saved to $ENV_FILE"
else
  echo "→ Skipping HF_TOKEN (unauthenticated requests)"
fi

echo ""

echo "Copying service files..."
sudo cp "$SCRIPT_DIR/pgvector.service" "$SERVICE_DIR/"
sudo cp "$SCRIPT_DIR/arxiv-agent.service" "$SERVICE_DIR/"
sudo cp "$SCRIPT_DIR/arxiv-agent-daily-job.service" "$SERVICE_DIR/"

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling services..."
sudo systemctl enable pgvector.service
sudo systemctl enable arxiv-agent.service
sudo systemctl enable arxiv-agent-daily-job.service

echo "Done. Services will start on boot:"
echo "  1. pgvector (vector database)"
echo "  2. arxiv-agent (LangGraph agent server)"
echo "  3. arxiv-agent-daily-job (embedding + bookmarks, then auto-stop)"
echo ""
echo "Check status: systemctl status pgvector arxiv-agent arxiv-agent-daily-job"
