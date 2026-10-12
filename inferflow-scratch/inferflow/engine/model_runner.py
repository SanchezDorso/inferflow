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
        # rank 0 持有所有 worker 的 Event 列表；worker 只持有自己的 Event
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
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        if self.is_cuda:
            torch.cuda.synchronize()
        dist.destroy_process_group()

    # ---------------- 多进程 RPC（Step8） ----------------

    def loop(self):
        """worker 主循环：不断读取 rank 0 的指令并执行，直到收到 exit。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        TODO(Step8): worker 端：event.wait() 等待唤醒；
          共享内存前 4 字节为 payload 长度 n（little endian），之后 n 字节为 pickle([method_name, *args])；
          读完后 event.clear()，返回 (method_name, args)
        """
        raise NotImplementedError

    def write_shm(self, method_name: str, *args):
        """TODO(Step8): rank 0 端：pickle → 写长度 + 数据 → 对每个 worker 的 event.set()"""
        raise NotImplementedError

    def call(self, method_name: str, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        return getattr(self, method_name)(*args)

    # ---------------- 显存规划（Step5） ----------------

    def warmup_model(self):
        """
        TODO(Step5, 仅 CUDA): 用最大 prefill 批跑一次前向，得到激活峰值显存，用于推算 KV cache 容量。
          empty_cache + reset_peak_memory_stats；
          构造 num_seqs = min(max_num_batched_tokens // max_model_len, max_num_seqs) 条长度为 max_model_len 的假序列；
          self.run(seqs, True)；empty_cache。
          CPU 下直接 return。
        """
        if not self.is_cuda:
            return
        raise NotImplementedError

    def allocate_kv_cache(self):
        """
        TODO(Step5):
          每块字节数 block_bytes = 2(K,V) * num_layers * block_size * num_kv_heads(本 rank) * head_dim * dtype.itemsize
          CUDA：num_blocks = (total * gpu_memory_utilization - used - peak + current) // block_bytes
                其中 free,total = torch.cuda.mem_get_info()，used = total - free，
                peak/current 来自 torch.cuda.memory_stats() 的 allocated_bytes.all.peak / current；
                若 config.num_kvcache_blocks > 0，再与之取 min（便于测试抢占）
          CPU ：config.num_kvcache_blocks > 0 则用之，否则 4 * ceil(max_model_len / block_size)
          Step8：多卡时各 rank 结果 all_reduce(MIN)，保证 block id 在所有 rank 都合法
          写回 config.num_kvcache_blocks（Scheduler 会读取）
          self.kv_cache = empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
          遍历 self.model.modules()，给每个带 k_cache/v_cache 属性的 Attention 按层号注入 kv_cache[0/1, layer_id]
        """
        raise NotImplementedError

    # ---------------- 输入准备（Step5） ----------------

    def _to_device(self, data, dtype) -> torch.Tensor:
        t = torch.tensor(data, dtype=dtype, device="cpu", pin_memory=self.is_cuda)
        return t.to(self.device, non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        # TODO(Step5): 把各序列 block_table 用 -1 右侧补齐成矩阵 [B, max_blocks]（int32）
        raise NotImplementedError

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        TODO(Step5): 把多条序列拼成一维 varlen batch，只计算未命中前缀缓存的部分：
          input_ids / positions：每条序列 [num_cached_tokens, len) 区间
          cu_seqlens_q：累加 len - num_cached_tokens；cu_seqlens_k：累加 len
          max_seqlen_q / max_seqlen_k
          slot_mapping：对 [num_cached_blocks, num_blocks) 的每个块，列出本次要写入的 slot
                        （最后一块只有 last_block_num_tokens 个）；warmup 时 block_table 为空则跳过
          Step6：若 cu_seqlens_k[-1] > cu_seqlens_q[-1]（有前缀命中），还需要 block_tables
          set_context(True, ...)，返回 (input_ids int64, positions int64)
        """
        raise NotImplementedError

    def prepare_decode(self, seqs: list[Sequence]):
        """
        TODO(Step5): 每条序列 1 个 token：
          input_ids = last_token，positions = len-1，context_lens = len，
          slot_mapping = block_table[-1] * block_size + last_block_num_tokens - 1，
          block_tables = prepare_block_tables(seqs)
          set_context(False, ...)，返回 (input_ids, positions)
        """
        raise NotImplementedError

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        return self._to_device([seq.temperature for seq in seqs], torch.float32)

    # ---------------- 前向 ----------------

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """
        TODO(Step5): eager 路径：self.model.compute_logits(self.model(input_ids, positions))
        TODO(Step9): decode 且非 enforce_eager 且 batch <= 最大档位时，走 CUDA Graph：
          选 >= bs 的最小档位的 graph；把 input_ids/positions/slot_mapping/context_lens/block_tables
          拷进 graph_vars 的前 bs 行（slot_mapping 先 fill_(-1)、context_lens 先 zero_()）；
          graph.replay()；对 graph_vars["outputs"][:bs] 计算 logits
        """
        raise NotImplementedError

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        TODO(Step9): decode 每步计算量小，kernel launch 开销占比高，按若干档 batch size 预先录制 CUDA Graph。
          max_bs = min(max_num_seqs, 512)；max_num_blocks = ceil(max_model_len / block_size)
          分配静态输入：input_ids, positions, slot_mapping(填 -1), context_lens, block_tables, outputs[max_bs, hidden]
          self.graph_bs = [1, 2, 4, 8, 16, 32, ..., max_bs]
          从大到小逐档：set_context(False, 静态张量[:bs])；先 eager 跑一次 warmup，
            再 with torch.cuda.graph(graph, self.graph_pool): outputs[:bs] = self.model(...)
            首个 graph 的 pool 供后续共享（graph.pool()）
          保存 self.graphs[bs]、self.graph_vars = dict(...)
        """
        raise NotImplementedError
