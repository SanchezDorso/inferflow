#!/usr/bin/env bash
# 下载模型权重到本地目录。
#   bash scripts/download_model.sh                              # Qwen/Qwen3-0.6B -> ~/huggingface/Qwen3-0.6B
#   bash scripts/download_model.sh Qwen/Qwen3-1.7B ~/huggingface/Qwen3-1.7B
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_model.sh   # 国内镜像
set -euo pipefail

REPO="${1:-Qwen/Qwen3-0.6B}"
DEST="${2:-$HOME/huggingface/$(basename "$REPO")}"
PY="${PY:-python}"

mkdir -p "$DEST"
"$PY" - "$REPO" "$DEST" <<'EOF'
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print(f"downloaded {repo} -> {path}")
EOF
