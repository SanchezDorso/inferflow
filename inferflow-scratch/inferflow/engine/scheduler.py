from collections import deque

from inferflow.config import Config
from inferflow.engine.block_manager import BlockManager
from inferflow.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    两阶段调度：
    1. Prefill：严格 FIFO，按到达顺序尽可能多地接纳 waiting 队列头部的请求，
       受 max_num_seqs / max_num_batched_tokens / 空闲 KV 块三重约束；队头放不下就停止（不允许插队）；
    2. Decode：没有可 prefill 的请求时，对 running 队列做一步解码。
       若 KV 块不足，从 running 队尾（最晚进入的请求）开始抢占：
       释放其全部 KV 块、状态回滚为 WAITING 并插回 waiting 队首，
       之后以 recompute 的方式（prompt + 已生成 token 一起重新 prefill）恢复。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = set(config.eos)
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size, config.enable_prefix_caching
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0
        self.num_prompt_tokens = 0
        self.num_prefix_hit_tokens = 0

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        assert len(seq) < self.max_model_len, f"prompt too long: {len(seq)} >= max_model_len={self.max_model_len}"
        num_blocks_needed = (self.max_model_len + self.block_manager.block_size - 1) // self.block_manager.block_size
        assert num_blocks_needed <= len(self.block_manager.blocks), "KV cache too small to hold max_model_len"
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        返回 (本步要执行的序列, 是否为 prefill)。
        TODO(Step2) Prefill 阶段：
          while waiting 非空 且 未达 max_num_seqs:
            队头 seq 若 token 预算（num_batched_tokens + len(seq)）超限或 can_allocate 失败 → break
            allocate；状态改 RUNNING、in_prefill=True；移入 running
            统计：num_batched_tokens += len(seq) - seq.num_cached_tokens；
                  首次 prefill（num_completion_tokens == 0）时累计 num_prompt_tokens / num_prefix_hit_tokens
          有 prefill 序列则直接返回 (seqs, True)
        TODO(Step7) Decode 阶段（含抢占）：
          依次从 running 队头取 seq：
            while not can_append(seq):
              running 还有别的序列 → preempt(running 队尾)
              否则 → preempt(seq) 自己，跳出
            （用 while-else：没被抢占才）may_append、in_prefill=False、加入本步列表
          最后把本步序列按原顺序放回 running 队头（extendleft(reversed(...))）
        """
        raise NotImplementedError

    def preempt(self, seq: Sequence):
        # TODO(Step7): 回滚：状态改 WAITING，释放全部块，插回 waiting 队首；num_preemptions += 1
        raise NotImplementedError

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        TODO(Step2): 为每条序列追加采样出的 token；
          满足 (未 ignore_eos 且 token 属于 eos) / 生成数达到 max_tokens / 长度达到 max_model_len 之一则结束：
          状态改 FINISHED、释放块、从 running 移除。
        """
        raise NotImplementedError
