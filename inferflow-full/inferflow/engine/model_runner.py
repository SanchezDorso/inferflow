import os
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import numpy as np
import torch
import torch.distributed as dist

from inferflow.config import Config
from inferflow.engine.sequence import Sequence
from inferflow.layers.flashinfer_decode import flashinfer_decode_enabled, plan_decode
from inferflow.layers.sampler import Sampler
from inferflow.models.qwen3 import Qwen3ForCausalLM
from inferflow.utils.context import get_context, reset_context, set_context
from inferflow.utils.loader import load_model

SHM_SIZE = 1 << 24


class ModelRunner:
    """
    每个 TP rank 一个 ModelRunner（rank 0 在主进程，其余在子进程）。

    rank 0 把 (方法名, 参数) pickle 后写入共享内存并通过 Event 唤醒 worker，
    所有 rank 执行同一方法，层内的 all_reduce / gather 由 NCCL 完成，
    最终只有 rank 0 做采样并把 token 返回给调度器。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.device = torch.device(config.device, rank) if config.device == "cuda" else torch.device("cpu")
        self.is_cuda = self.device.type == "cuda"
        Sequence.block_size = self.block_size

        dist.init_process_group(
            "nccl" if self.is_cuda else "gloo",
            f"tcp://127.0.0.1:{config.dist_port}",
            world_size=self.world_size,
            rank=rank,
        )
        if self.is_cuda:
            from inferflow.utils.affinity import bind_current_process_to_gpu, physical_gpu_index

            bind_current_process_to_gpu(physical_gpu_index(rank))
            torch.cuda.set_device(rank)
        else:
            torch._dynamo.config.disable = True
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.torch_dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # warmup 的 prefill 会走到 run_model，属性要先存在。KV 页在 allocate 之后才有。
        self.flashinfer_decode = self.is_cuda and flashinfer_decode_enabled()
        self._fi_attn = None
        self.warmup_model()
        self.allocate_kv_cache()
        if self.flashinfer_decode:
            self._fi_attn = next(m for m in self.model.modules() if hasattr(m, "k_cache") and m.k_cache.numel())
        self.graph_bs = [1]
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            shm_name = f"inferflow_{config.dist_port}"
            if rank == 0:
                try:
                    SharedMemory(name=shm_name).unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name=shm_name, create=True, size=SHM_SIZE)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        self.kv_cache = None
        self.model = None
        self.sampler = None
        if not self.enforce_eager:
            self.graphs = None
            self.graph_pool = None
            self.graph_vars = None
        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        dist.destroy_process_group()

    # ---------------- 多进程 RPC ----------------

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name: str, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= SHM_SIZE, f"RPC payload too large: {n} bytes"
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name: str, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        return getattr(self, method_name)(*args)

    # ---------------- 显存规划 ----------------

    def warmup_model(self):
        """用最大 prefill 批跑一次前向，得到激活峰值显存，用于后续推算 KV cache 容量。"""
        if not self.is_cuda:
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = max(1, min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs))
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        # 前缀缓存关闭时 KV 用 fp8。同样的显存放两倍 token，decode 每次仍读满整张 cache，
        # 但每个 token 只有一半字节，同样的步长能吐出大约两倍的 token。
        self.fp8_kv = (
            self.flashinfer_decode
            and not config.enable_prefix_caching
            and os.environ.get("INFERFLOW_FP8_KV", "1") != "0"
        )
        dtype = torch.float8_e4m3fn if self.fp8_kv else config.torch_dtype
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype.itemsize

        if self.is_cuda:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            stats = torch.cuda.memory_stats()
            peak, current = stats["allocated_bytes.all.peak"], stats["allocated_bytes.all.current"]
            num_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
            if config.num_kvcache_blocks > 0:
                num_blocks = min(num_blocks, config.num_kvcache_blocks)
        else:
            max_blocks_per_seq = (config.max_model_len + self.block_size - 1) // self.block_size
            num_blocks = config.num_kvcache_blocks if config.num_kvcache_blocks > 0 else 4 * max_blocks_per_seq
        # 各 rank 剩余显存可能不同，取最小值保证 block id 在所有 rank 上都合法
        t = torch.tensor([num_blocks], dtype=torch.int64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        config.num_kvcache_blocks = int(t.item())
        assert config.num_kvcache_blocks > 0, "not enough GPU memory for KV cache"

        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim,
            dtype=dtype, device=self.device,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        if self.rank == 0:
            kind = "fp8 e4m3" if self.fp8_kv else str(dtype).replace("torch.", "")
            print(
                f"[InferFlow] KV cache: {config.num_kvcache_blocks} blocks x {self.block_size} tokens "
                f"({config.num_kvcache_blocks * block_bytes / 2**30:.2f} GiB per rank, {kind})"
            )

    # ---------------- 输入准备 ----------------

    def _to_device(self, data, dtype) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            # torch.tensor(..., pin_memory=True) 不能直接包 numpy。
            np_dtype = {
                torch.int32: np.int32,
                torch.int64: np.int64,
                torch.float32: np.float32,
            }[dtype]
            array = np.ascontiguousarray(data, dtype=np_dtype)
            t = torch.from_numpy(array)
            if self.is_cuda:
                t = t.pin_memory().to(self.device, non_blocking=True)
            return t
        t = torch.tensor(data, dtype=dtype, device="cpu", pin_memory=self.is_cuda)
        return t.to(self.device, non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        table = np.full((len(seqs), max_len), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            n = len(seq.block_table)
            if n:
                table[i, :n] = seq.block_table
        return self._to_device(table, torch.int32)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping = [], [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q = max_seqlen_k = 0
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # Decode 只算最后一个词，历史 KV 从分页缓存读。Prefill 从尚未命中的位置算到结尾。
            q_start = seq.num_cached_tokens if seq.in_prefill else seqlen - 1
            if seq.in_prefill:
                input_ids.extend(seq[q_start:])
            else:
                input_ids.append(seq.last_token)
            positions.extend(range(q_start, seqlen))
            seqlen_q = seqlen - q_start
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            if seq.in_prefill:
                for i in range(seq.num_cached_blocks, seq.num_blocks):
                    start = seq.block_table[i] * self.block_size
                    end = start + (self.block_size if i != seq.num_blocks - 1 else seq.last_block_num_tokens)
                    slot_mapping.extend(range(start, end))
            else:
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        # 存在前缀缓存命中时 K 比 Q 长，需要通过 block_table 从 KV cache 读取完整 K/V
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        set_context(
            True,
            cu_seqlens_q=self._to_device(cu_seqlens_q, torch.int32),
            cu_seqlens_k=self._to_device(cu_seqlens_k, torch.int32),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=self._to_device(slot_mapping, torch.int32),
            block_tables=block_tables,
        )
        return self._to_device(input_ids, torch.int64), self._to_device(positions, torch.int64)

    def prepare_decode(self, seqs: list[Sequence]):
        n = len(seqs)
        input_ids = np.empty(n, dtype=np.int64)
        positions = np.empty(n, dtype=np.int64)
        slot_mapping = np.empty(n, dtype=np.int32)
        context_lens = np.empty(n, dtype=np.int32)
        max_len = max(len(seq.block_table) for seq in seqs)
        table = np.full((n, max_len), -1, dtype=np.int32)
        bs = self.block_size
        for i, seq in enumerate(seqs):
            input_ids[i] = seq.last_token
            length = len(seq)
            positions[i] = length - 1
            context_lens[i] = length
            bt = seq.block_table
            table[i, : len(bt)] = bt
            slot_mapping[i] = bt[-1] * bs + seq.last_block_num_tokens - 1
        set_context(
            False,
            slot_mapping=self._to_device(slot_mapping, torch.int32),
            context_lens=self._to_device(context_lens, torch.int32),
            block_tables=self._to_device(table, torch.int32),
        )
        return self._to_device(input_ids, torch.int64), self._to_device(positions, torch.int64)

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        return self._to_device([seq.temperature for seq in seqs], torch.float32)

    # ---------------- 前向 ----------------

    def _plan_flashinfer(self, block_tables: torch.Tensor, context_lens: torch.Tensor) -> None:
        attn = self._fi_attn
        page = self.block_size
        max_pages = (self.config.max_model_len + page - 1) // page
        plan_decode(
            block_tables,
            context_lens,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            page,
            self.config.torch_dtype,
            attn.k_cache.dtype,
            attn.scale,
            max_pages,
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        bs = input_ids.size(0)
        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            if self.flashinfer_decode and not is_prefill:
                context = get_context()
                self._plan_flashinfer(context.block_tables, context.context_lens)
            return self.model.compute_logits(self.model(input_ids, positions))
        context = get_context()
        bucket = next(x for x in self.graph_bs if x >= bs)
        graph = self.graphs[bucket]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        if self.flashinfer_decode:
            if bucket > bs:
                graph_vars["context_lens"][bs:bucket] = 1
                graph_vars["block_tables"][bs:bucket].zero_()
            self._plan_flashinfer(graph_vars["block_tables"][:bucket], graph_vars["context_lens"][:bucket])
        graph.replay()
        if self.graph_logits:
            return graph_vars["logits"][:bs]
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = None
        if self.rank == 0:
            # 全贪心时不要先做一次 GPU all() 再同步，否则每个解码步多一次往返。
            if all(seq.temperature == 0.0 for seq in seqs):
                token_ids = logits.argmax(dim=-1).tolist()
            else:
                token_ids = self.sampler(logits, self.prepare_sample(seqs)).tolist()
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """decode 阶段每步计算量小、kernel launch 开销占比高，按若干档 batch size 预先录制 CUDA Graph。"""
        config = self.config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        # decode 的 flash kernel 用 context_lens - 1 作为写入位置，捕获时不能是负数。
        context_lens = torch.ones(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, config.hf_config.hidden_size)
        # 单卡时把 lm_head 一并录进图。多卡 gather 留在图外，避免 NCCL 进 CUDA Graph。
        self.graph_logits = self.world_size == 1
        vocab = config.hf_config.vocab_size
        logits = torch.zeros(max_bs, vocab, dtype=config.torch_dtype) if self.graph_logits else None
        # 16 一档时，常驻大约 40 条会被抬到 48，多读 20% 的 KV。
        # 长上下文 decode 吃带宽，这一档空白直接变成墙钟。8 一档盖住这条工作负载的常驻 batch。
        self.graph_bs = sorted(
            {bs for bs in [1, 2, 4, 8, *range(16, min(max_bs, 128) + 1, 8), *range(144, max_bs + 1, 16)] if 0 < bs <= max_bs}
            | {max_bs}
        )
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            if self.flashinfer_decode:
                self._plan_flashinfer(block_tables[:bs], context_lens[:bs])
            hidden = self.model(input_ids[:bs], positions[:bs])
            if self.graph_logits:
                logits[:bs] = self.model.compute_logits(hidden)
            else:
                outputs[:bs] = hidden
            with torch.cuda.graph(graph, self.graph_pool):
                hidden = self.model(input_ids[:bs], positions[:bs])
                if self.graph_logits:
                    logits[:bs] = self.model.compute_logits(hidden)
                else:
                    outputs[:bs] = hidden
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            logits=logits,
        )
