from copy import copy
from enum import Enum, auto
from itertools import count

from inferflow.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    一条推理请求。token_ids = prompt + 已生成的 token。
    block_table[i] 是第 i 个逻辑块（token 区间 [i*block_size, (i+1)*block_size)）对应的物理 KV 块 id。
    """

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        sampling_params = sampling_params or SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        # 前缀缓存命中的 token 数（总是 block_size 的整数倍）
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        # 被抢占后重新 prefill 时 worker 需要完整 token 序列，decode 时只需 last_token
        self.in_prefill = True
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self) -> int:
        # TODO(Step1): 前缀缓存命中的块数
        raise NotImplementedError

    @property
    def num_blocks(self) -> int:
        # TODO(Step1): 容纳 num_tokens 需要的块数（向上取整）
        raise NotImplementedError

    @property
    def last_block_num_tokens(self) -> int:
        # TODO(Step1): 最后一个块里实际有多少 token（1 ~ block_size）
        raise NotImplementedError

    def block(self, i: int) -> list[int]:
        # TODO(Step1): 返回第 i 个逻辑块的 token_ids
        raise NotImplementedError

    def append_token(self, token_id: int):
        # TODO(Step1): 追加新 token，同步更新 last_token / num_tokens
        raise NotImplementedError

    # TODO(Step8): 跨进程传给 TP worker 时只序列化必要字段，降低共享内存通信量。
    #   worker 需要：num_tokens, num_prompt_tokens, num_cached_tokens, block_table, in_prefill，
    #   以及 prefill 时的完整 token_ids / decode 时的 last_token。
    #   Step8 之前可以不实现（pickle 会默认序列化全部字段）。
    # def __getstate__(self): ...
    # def __setstate__(self, state): ...
