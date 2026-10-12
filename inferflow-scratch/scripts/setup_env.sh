#!/usr/bin/env bash
# 在 GPU 机器上一键搭建 InferFlow 运行环境。
#
# 用法：
#   bash scripts/setup_env.sh                 # 默认 venv 路径 ./.venv
#   VENV=/path/to/venv bash scripts/setup_env.sh
#
# 可选环境变量：
#   PYTHON         用于创建 venv 的解释器（默认 python3，需 3.10~3.12）
#   TORCH_VERSION  默认 2.7.1
#   TORCH_CUDA     默认 cu126（可选 cu118 / cu124 / cu126 / cu128，需与驱动匹配）
#   FLASH_ATTN     默认 2.8.3
#   PIP_INDEX_URL  国内可设为 https://pypi.tuna.tsinghua.edu.cn/simple
#   GH_PROXY       GitHub 加速前缀，例如 https://ghfast.top/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
FLASH_ATTN="${FLASH_ATTN:-2.8.3}"
GH_PROXY="${GH_PROXY:-}"

log() { echo -e "\033[32m[setup]\033[0m $*"; }

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: 请在 GPU 机器上运行"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ ! -x "$VENV/bin/python" ]; then
    log "create venv at $VENV"
    "$PYTHON" -m venv "$VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
"$PIP" install -q --upgrade pip wheel setuptools

log "install torch==$TORCH_VERSION ($TORCH_CUDA)"
"$PIP" install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

log "install python deps"
"$PIP" install "transformers>=4.51,<5" safetensors xxhash numpy tqdm pytest huggingface_hub

log "install flash-attn==$FLASH_ATTN (prebuilt wheel)"
WHEEL_NAME="$("$PY" - "$FLASH_ATTN" <<'EOF'
import sys, torch
fa = sys.argv[1]
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cuda_major = torch.version.cuda.split(".")[0]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"flash_attn-{fa}+cu{cuda_major}torch{torch_mm}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
EOF
)"
WHEEL_URL="${GH_PROXY}https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN}/${WHEEL_NAME}"
if "$PIP" install "$WHEEL_URL"; then
    log "flash-attn installed from $WHEEL_NAME"
else
    log "prebuilt wheel not found, building from source (可能需要 10~30 分钟)"
    MAX_JOBS="${MAX_JOBS:-8}" "$PIP" install "flash-attn==$FLASH_ATTN" --no-build-isolation
fi

log "install inferflow (editable)"
"$PIP" install -e "$ROOT" --no-deps

log "verify"
"$PY" "$ROOT/scripts/check_env.py"
log "done. 激活环境: source $VENV/bin/activate"
