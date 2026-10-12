import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from inferflow.config import Config
from inferflow.engine.sequence import Sequence
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
            torch.cuda.set_device(rank)
        else:
            torch._dynamo.config.disable = True
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.torch_dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
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
        dtype = config.torch_dtype
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
            print(
                f"[InferFlow] KV cache: {config.num_kvcache_blocks} blocks x {self.block_size} tokens "
                f"({config.num_kvcache_blocks * block_bytes / 2**30:.2f} GiB per rank)"
            )

    # ---------------- 输入准备 ----------------

    def _to_device(self, data, dtype) -> torch.Tensor:
        t = torch.tensor(data, dtype=dtype, device="cpu", pin_memory=self.is_cuda)
        return t.to(self.device, non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return self._to_device(block_tables, torch.int32)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping = [], [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q = max_seqlen_k = 0
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(range(seq.num_cached_tokens, seqlen))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                end = start + (self.block_size if i != seq.num_blocks - 1 else seq.last_block_num_tokens)
                slot_mapping.extend(range(start, end))
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
        input_ids, positions, slot_mapping, context_lens = [], [], [], []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        set_context(
            False,
            slot_mapping=self._to_device(slot_mapping, torch.int32),
            context_lens=self._to_device(context_lens, torch.int32),
            block_tables=self.prepare_block_tables(seqs),
        )
        return self._to_device(input_ids, torch.int64), self._to_device(positions, torch.int64)

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        return self._to_device([seq.temperature for seq in seqs], torch.float32)

    # ---------------- 前向 ----------------

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        bs = input_ids.size(0)
        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            return self.model.compute_logits(self.model(input_ids, positions))
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
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
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, config.hf_config.hidden_size)
        self.graph_bs = sorted({bs for bs in [1, 2, 4, 8, *range(16, max_bs + 1, 16)] if bs <= max_bs} | {max_bs})
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
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
        )
