"""按请求把推理分到多张卡。

Qwen3-0.6B 放得进单卡。张量并行每层都要 all-reduce，PCIe 上的通信比省下的计算还多，
卡越多越慢。这里每张卡留一份完整模型，各自跑连续批处理。主进程按已分配的 prompt token 数把请求交给当前最闲的卡。
"""

import multiprocessing as mp
import os
import subprocess
from queue import Empty


def _gpu_count() -> int:
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def _worker(rank, model, engine_kwargs, request_q, result_q, barrier, pull_per_step):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from inferflow.utils.affinity import bind_current_process_to_gpu

    bind_current_process_to_gpu(rank)
    from inferflow import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(model, **engine_kwargs)
        llm.generate(
            [[1, 2, 3, 4]],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        barrier.wait()
        index_of: dict[int, int] = {}
        shutdown = False
        max_inflight = engine_kwargs.get("max_num_seqs", 512)

        def emit(finished):
            for seq_id, token_ids in finished:
                result_q.put((index_of.pop(seq_id), token_ids, rank))

        def admit(item) -> bool:
            if item is None:
                return True
            origin, token_ids, sampling_params = item
            seq_id = llm.add_request(token_ids, sampling_params)
            index_of[seq_id] = origin
            return False

        while True:
            if shutdown and llm.is_finished():
                break
            pulled = 0
            while not shutdown and len(index_of) < max_inflight and pulled < pull_per_step:
                if not index_of and pulled == 0:
                    item = request_q.get()
                else:
                    try:
                        item = request_q.get_nowait()
                    except Empty:
                        break
                if admit(item):
                    shutdown = True
                    break
                pulled += 1
            if llm.is_finished():
                continue
            finished, _ = llm.step()
            emit(finished)
    except BaseException as exc:
        result_q.put(("error", rank, repr(exc)))
        barrier.abort()
        raise
    finally:
        if llm is not None:
            llm.exit()


class DataParallelLLM:
    """一张卡一个引擎。请求进公共队列，哪张卡空闲就由哪张卡的调度器接走。"""

    def __init__(self, model: str, data_parallel_size: int | None = None, **kwargs):
        available = _gpu_count()
        if data_parallel_size is None:
            data_parallel_size = available
        if data_parallel_size < 1 or data_parallel_size > available:
            raise ValueError(f"data_parallel_size={data_parallel_size}，当前可见 GPU 为 {available}")
        if kwargs.get("tensor_parallel_size", 1) != 1:
            raise ValueError("数据并行的每张卡各自持有完整模型，tensor_parallel_size 必须为 1")
        self.model = model
        self.data_parallel_size = data_parallel_size
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self._exited = False

        ctx = mp.get_context("spawn")
        self._request_qs = [ctx.Queue() for _ in range(data_parallel_size)]
        self._result_q = ctx.Queue()
        self._barrier = ctx.Barrier(data_parallel_size + 1)
        port_base = 25000 + (os.getpid() % 2000)
        self._procs = []
        # 子进程在 import torch 时就会初始化 CUDA。可见卡必须在 Process.start 之前写进环境，
        # 写在 worker 函数里已经晚了，两张卡会抢同一块 GPU。
        saved_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for rank in range(data_parallel_size):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
                engine_kwargs = {
                    k: v for k, v in kwargs.items() if k != "data_parallel_size"
                }
                engine_kwargs["tensor_parallel_size"] = 1
                engine_kwargs["dist_port"] = port_base + rank
                pull_per_step = engine_kwargs.get("max_num_seqs", 512)
                proc = ctx.Process(
                    target=_worker,
                    args=(rank, model, engine_kwargs, self._request_qs[rank], self._result_q, self._barrier, pull_per_step),
                )
                proc.start()
                self._procs.append(proc)
        finally:
            if saved_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_visible
        try:
            self._barrier.wait(timeout=600)
        except Exception:
            err = None
            try:
                err = self._result_q.get(timeout=2)
            except Empty:
                pass
            self.exit()
            raise RuntimeError(f"data-parallel workers failed to start: {err}")

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params,
        use_tqdm: bool = True,
        decode: bool = True,
    ) -> list[dict]:
        del use_tqdm
        if self._exited:
            raise RuntimeError("engine already exited")
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params 与 prompts 数量不一致")
        encoded = [self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt) for prompt in prompts]
        assigned_tokens = [0] * self.data_parallel_size
        # 前缀缓存是每张卡私有的。相同的满块前缀固定分到同一张卡，负载已经更高时改走最闲的卡。
        block_size = 256
        sticky: dict[tuple[int, ...], int] = {}
        for origin, (token_ids, params) in enumerate(zip(encoded, sampling_params)):
            prefix = tuple(token_ids[:block_size]) if len(token_ids) >= block_size else None
            least = min(range(self.data_parallel_size), key=lambda i: (assigned_tokens[i], i))
            rank = least
            if prefix is not None and prefix in sticky:
                pinned = sticky[prefix]
                if assigned_tokens[pinned] <= assigned_tokens[least] + len(token_ids):
                    rank = pinned
            if prefix is not None:
                sticky.setdefault(prefix, rank)
            assigned_tokens[rank] += len(token_ids)
            self._request_qs[rank].put((origin, token_ids, params))
        pending = len(encoded)
        token_ids_out: list[list[int] | None] = [None] * pending
        counts = [0] * self.data_parallel_size
        while pending:
            item = self._result_q.get(timeout=600)
            if item[0] == "error":
                raise RuntimeError(f"data-parallel worker failed: {item}")
            origin, completion, rank = item
            token_ids_out[origin] = completion
            counts[rank] += 1
            pending -= 1
        self.last_counts = counts
        return [
            {
                "text": self.tokenizer.decode(completion) if decode else "",
                "token_ids": completion,
            }
            for completion in token_ids_out
        ]

    def exit(self):
        if self._exited:
            return
        self._exited = True
        for request_q in self._request_qs:
            request_q.put(None)
        for proc in self._procs:
            proc.join(timeout=120)
            if proc.is_alive():
                proc.terminate()
