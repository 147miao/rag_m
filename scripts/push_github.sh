#!/usr/bin/env bash
# 推送仓库到 GitHub，用法：scripts/push_github.sh git@github.com:<user>/<repo>.git
set -euo pipefail
URL="${1:?用法: scripts/push_github.sh git@github.com:<user>/<repo>.git}"
cd "$(dirname "$0")/.."
git remote remove origin 2>/dev/null || true
git remote add origin "$URL"
git push -u origin main
echo "pushed: $(git log --oneline -1)"