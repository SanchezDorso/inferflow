from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # temperature == 0 表示贪心解码
        assert self.temperature >= 0.0
        assert self.max_tokens > 0
