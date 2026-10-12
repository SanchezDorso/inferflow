#!/usr/bin/env bash
# 按步骤运行对应测试：bash scripts/check_step.sh 3
# 步骤说明见 README.md「实现路线」
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python}"
STEP="${1:?usage: check_step.sh <step 1-9|all>}"

run() { echo "+ pytest $*"; "$PY" -m pytest -q "$@"; }

case "$STEP" in
    1) run tests/test_block_manager.py -k "allocate_and_free or disabled" ;;
    2) run tests/test_block_manager.py tests/test_scheduler.py \
           -k "allocate_and_free or disabled or may_append or fifo or blocked or max_num_seqs or decode_after or eos" ;;
    3) run tests/test_tp_layers.py ;;
    4) run tests/test_model_forward.py ;;
    5) run tests/test_engine_cpu.py -k batch_greedy ;;
    6) run tests/test_block_manager.py && run tests/test_engine_cpu.py -k prefix ;;
    7) run tests/test_scheduler.py && run tests/test_engine_cpu.py -k preemption ;;
    8) run tests/test_engine_cpu.py -k tensor_parallel ;;
    9) "$PY" scripts/check_env.py && run tests/test_qwen3_gpu.py -s ;;
    all) run tests ;;
    *) echo "unknown step: $STEP"; exit 1 ;;
esac
