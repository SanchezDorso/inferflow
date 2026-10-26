from collections import deque

from inferflow.config import Config
from inferflow.engine.block_manager import BlockManager
from inferflow.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    连续批处理：每一步可以同时包含 Decode 和 Prefill。

    先给 running 里的请求各排 1 个 decode token，保证已经在生成的请求不被新提示堵住。
    剩下的 token 预算再按 FIFO 接纳 waiting：队头放不下就停，后面的短请求不能插队。
    本步刚被抢占的请求留在 waiting，下一拍再和其余 Decode 一起重算。
    显存不足时从 running 队尾抢占，已生成的 token 保留，之后整段重新 Prefill。
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
        preempted: set[int] = set()

        # 先排 Decode。本步刚抢占的请求留在 waiting，避免把刚腾出的块立刻占回去。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            if num_batched_tokens + 1 > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)
                    preempted.add(victim.seq_id)
                else:
                    self.preempt(seq)
                    preempted.add(seq.seq_id)
                    break
            else:
                num_batched_tokens += 1
                self.block_manager.may_append(seq)
                seq.in_prefill = False
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))

        # 剩余预算按 FIFO 接 Prefill，和上面的 Decode 放进同一批。
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if seq.seq_id in preempted:
                break
            _, cached_tokens = self.block_manager.preview(seq)
            new_tokens = len(seq) - cached_tokens
            if num_batched_tokens + new_tokens > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
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
        return scheduled_seqs, any(seq.in_prefill for seq in scheduled_seqs)

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
