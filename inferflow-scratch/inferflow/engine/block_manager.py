from collections import deque

import numpy as np
import xxhash

from inferflow.engine.sequence import Sequence


class Block:
    """一个物理 KV 块的元信息（真正的 KV 数据在 ModelRunner.kv_cache 里）。"""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    KV cache 物理块管理 + 前缀缓存。

    数据结构：
      blocks            所有物理块
      free_block_ids    空闲块队列（从队头分配、释放时追加到队尾 → 天然 LRU）
      used_block_ids    正在被引用的块
      hash_to_block_id  满块的 hash -> 物理块 id，用于前缀复用

    关键设计：
      - 只有「满块」才计算 hash，且 hash 链式依赖前一块：h_i = xxh64(h_{i-1} || tokens_i)，
        因此 hash 相同 ⇔ 从序列开头到该块结束的前缀完全相同；
      - 块被释放（ref_count 归零）后不清空 hash/内容，直到被重新分配，这样空闲块仍可被命中；
      - 命中判断需同时校验 block.hash == h 且 block.token_ids == token_ids，
        防止 hash_to_block_id 中的陈旧映射导致误命中（见 tests/test_block_manager.py）；
      - prompt 的最后一块总是重新计算，保证 prefill 至少有 1 个 token 产生 logits。
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        assert num_blocks > 0
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        # TODO(Step1): 把 block_id 从空闲队列移到 used 集合，reset 后返回
        raise NotImplementedError

    def _deallocate_block(self, block_id: int):
        # TODO(Step1): 把 block_id 从 used 集合放回空闲队列队尾
        raise NotImplementedError

    def can_allocate(self, seq: Sequence) -> bool:
        # TODO(Step1): 空闲块是否足够容纳整条序列（保守估计，不考虑前缀命中）
        raise NotImplementedError

    def allocate(self, seq: Sequence):
        """
        prefill 前为序列分配全部块，填充 seq.block_table。
        TODO(Step1): 先实现不带前缀缓存的版本：每个逻辑块取空闲队列队头的块；
          满块顺便计算链式 hash 并 block.update / 登记 hash_to_block_id（Step6 查找时要用）。
        TODO(Step6): 前缀缓存：
          逐块计算链式 hash（非满块 hash=-1），查 hash_to_block_id；
          一旦某块未命中，后续块全部视为未命中（cache_miss 粘滞）；
          命中时 seq.num_cached_tokens += block_size，
            若块仍在 used 中 → ref_count += 1；若块在空闲队列中 → 重新 _allocate_block；
          满块分配后调用 block.update(h, token_ids) 并登记 hash_to_block_id。
          enable_prefix_caching=False 时全部视为未命中。
        """
        raise NotImplementedError

    def deallocate(self, seq: Sequence):
        # TODO(Step1): 逆序遍历 block_table，ref_count -= 1，归零则释放；清空 block_table 与 num_cached_tokens
        raise NotImplementedError

    def can_append(self, seq: Sequence) -> bool:
        # TODO(Step2): decode 前判断：若 last_token 恰好落在新块的第一个位置（len % block_size == 1），需要 1 个空闲块
        raise NotImplementedError

    def may_append(self, seq: Sequence):
        """
        decode 前调用，为 last_token 的 KV 预留槽位。
        TODO(Step2):
          len % block_size == 1 → 分配新块追加到 block_table
          len % block_size == 0 → 最后一块刚好写满：用前一块的 hash 作 prefix 计算本块 hash，
                                  block.update 并登记 hash_to_block_id（Step6 的前缀复用依赖它）
          其它                   → 无事可做
        """
        raise NotImplementedError
