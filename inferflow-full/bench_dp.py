"""数据并行吞吐：每张卡一份完整模型，空闲卡从公共队列取请求。

用法：
  python bench_dp.py                 # 依次测 1 / 2 / 4 卡
  python bench_dp.py --dp 2
  python bench_dp.py --check         # 只核对 1 卡和 2 卡的 token 是否一致
"""

import argparse
import os
from time import perf_counter

from inferflow.engine.dp_engine import DataParallelLLM
from inferflow.sampling_params import SamplingParams

MODEL = os.path.expanduser("~/autodl-tmp/huggingface/Qwen3-0.6B")
if not os.path.isdir(MODEL):
    MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")

N = 128
PROMPT = 512
MAX_NEW = 128


def make_prompts(n=N, length=PROMPT):
    return [[(i * 997 + t) % 151936 for t in range(length)] for i in range(n)]


def run_dp(dp, prompts, max_new, eager):
    llm = DataParallelLLM(
        MODEL,
        data_parallel_size=dp,
        max_num_seqs=min(128, max(len(prompts), 1)),
        max_model_len=4096,
        max_num_batched_tokens=16384,
        enable_prefix_caching=True,
        enforce_eager=eager,
        gpu_memory_utilization=0.9,
    )
    try:
        sp = SamplingParams(temperature=0, max_tokens=max_new, ignore_eos=True)
        t0 = perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=False, decode=False)
        elapsed = perf_counter() - t0
        counts = list(llm.last_counts)
    finally:
        llm.exit()
    prompt_tokens = sum(len(prompt) for prompt in prompts)
    decode_tokens = sum(len(item["token_ids"]) for item in outputs)
    print(
        "SUMMARY "
        f"mode=dp dp={dp} wall_s={elapsed:.3f} "
        f"prompt_tokens={prompt_tokens} decode_tokens={decode_tokens} "
        f"e2e_tok_s={(prompt_tokens + decode_tokens) / elapsed:.1f} "
        f"decode_tok_s={decode_tokens / elapsed:.1f} "
        f"finished={len(outputs)} counts={counts}",
        flush=True,
    )
    return [item["token_ids"] for item in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        prompts = make_prompts(4, 32)
        first = run_dp(1, prompts, 8, eager=True)
        second = run_dp(1, prompts, 8, eager=True)
        print(f"CHECK dp1_repeat_match={first == second}", flush=True)
        split = run_dp(2, prompts, 8, eager=True)
        print(f"CHECK dp2_match_dp1={split == first}", flush=True)
        if first != second:
            raise SystemExit(1)
        return
    prompts = make_prompts()
    for dp in args.dp:
        print(f"\n===== DP {dp} =====", flush=True)
        run_dp(dp, prompts, MAX_NEW, eager=False)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
