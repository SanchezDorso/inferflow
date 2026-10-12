from collections import deque

from inferflow.config import Config
from inferflow.engine.block_manager import BlockManager
from inferflow.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    两阶段调度：
    1. Prefill：严格 FIFO，按到达顺序尽可能多地接纳 waiting 队列头部的请求，
       受 max_num_seqs / max_num_batched_tokens / 空闲 KV 块三重约束；
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
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0

        # max_num_seqs 限制的是 running 中的并发总数，而不是单次 prefill 的条数。
        # token 预算把 prefill 拆成多步时，后面的步也只能填满剩余槽位。
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            if seq.num_completion_tokens == 0:
                self.num_prompt_tokens += len(seq)
                self.num_prefix_hit_tokens += seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            seq.in_prefill = True
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                seq.in_prefill = False
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id in self.eos)
                or seq.num_completion_tokens >= seq.max_tokens
                or len(seq) >= self.max_model_len
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
