"""
吞吐基准：
  python bench.py                       # 512 条随机长度请求（连续批处理 + 抢占）
  python bench.py --scenario prefix     # 共享长 system prompt，观察前缀缓存命中率
  python bench.py --scenario long       # 接近 4096 的长上下文
  python bench.py --tp 2                # 张量并行
"""

import argparse
import os
import time
from random import randint, seed

from inferflow import LLM, SamplingParams


def build_requests(args):
    seed(0)
    n = args.num_seqs
    if args.scenario == "random":
        prompts = [[randint(0, 10000) for _ in range(randint(100, args.max_input_len))] for _ in range(n)]
        sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len)) for _ in range(n)]
    elif args.scenario == "prefix":
        system = [randint(0, 10000) for _ in range(2048)]
        prompts = [system + [randint(0, 10000) for _ in range(randint(16, 256))] for _ in range(n)]
        sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=128) for _ in range(n)]
    elif args.scenario == "long":
        prompts = [[randint(0, 10000) for _ in range(randint(3000, 3800))] for _ in range(n)]
        sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256) for _ in range(n)]
    else:
        raise ValueError(args.scenario)
    return prompts, sps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--scenario", choices=["random", "prefix", "long"], default="random")
    parser.add_argument("--num-seqs", type=int, default=512)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true")
    args = parser.parse_args()

    llm = LLM(
        args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.no_prefix_caching,
    )
    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)

    prompts, sps = build_requests(args)
    t = time.perf_counter()
    llm.generate(prompts, sps, use_tqdm=True)
    elapsed = time.perf_counter() - t

    num_output = sum(sp.max_tokens for sp in sps)
    num_input = sum(len(p) for p in prompts)
    print(f"\nscenario={args.scenario} tp={args.tp} seqs={len(prompts)}")
    print(f"input tokens : {num_input}")
    print(f"output tokens: {num_output}")
    print(f"time         : {elapsed:.2f}s")
    print(f"throughput   : {num_output / elapsed:.2f} output tok/s, {(num_input + num_output) / elapsed:.2f} total tok/s")
    print(f"stats        : {llm.stats()}")
    llm.exit()


if __name__ == "__main__":
    main()
