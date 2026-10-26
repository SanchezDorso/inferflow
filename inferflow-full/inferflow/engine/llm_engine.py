import atexit
import gc
from dataclasses import fields
from time import perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig

from inferflow.config import Config
from inferflow.engine.model_runner import ModelRunner
from inferflow.engine.scheduler import Scheduler
from inferflow.engine.sequence import Sequence
from inferflow.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config = Config(model, **{k: v for k, v in kwargs.items() if k in config_fields})
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self._collect_eos_ids(config.model)
        self.scheduler = Scheduler(config)
        self._exited = False
        atexit.register(self.exit)

    def _collect_eos_ids(self, model: str) -> tuple[int, ...]:
        eos = {self.tokenizer.eos_token_id}
        try:
            gen_eos = GenerationConfig.from_pretrained(model).eos_token_id
            eos.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
        except OSError:
            pass
        return tuple(sorted(e for e in eos if e is not None))

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """执行一次调度 + 前向。返回 (本步完成的序列, 处理的 token 数；prefill 为正、decode 为负)。"""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        # 混批里 Decode 只占 1 个词。正数只统计本步新算的 Prefill 词，纯 Decode 仍用负数。
        prefill_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs if seq.in_prefill)
        num_tokens = prefill_tokens if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts)
        seq_ids = [self.add_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            dt = perf_counter() - t
            if num_tokens > 0:
                prefill_throughput = num_tokens / dt
            elif num_tokens < 0:
                decode_throughput = -num_tokens / dt
            pbar.set_postfix(prefill=f"{int(prefill_throughput)}tok/s", decode=f"{int(decode_throughput)}tok/s")
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        return [
            {"text": self.tokenizer.decode(outputs[seq_id]), "token_ids": outputs[seq_id]} for seq_id in seq_ids
        ]

    def stats(self) -> dict:
        s = self.scheduler
        return {
            "num_kvcache_blocks": self.config.num_kvcache_blocks,
            "num_preemptions": s.num_preemptions,
            "prefix_hit_tokens": s.num_prefix_hit_tokens,
            "prompt_tokens": s.num_prompt_tokens,
            "prefix_hit_rate": s.num_prefix_hit_tokens / max(1, s.num_prompt_tokens),
        }
