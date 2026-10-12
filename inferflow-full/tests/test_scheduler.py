from types import SimpleNamespace

from inferflow.engine.scheduler import Scheduler
from inferflow.engine.sequence import Sequence, SequenceStatus
from inferflow.sampling_params import SamplingParams

BS = 256
EOS = 99999


def make_scheduler(num_blocks=16, max_num_seqs=512, max_num_batched_tokens=4096, max_model_len=1024):
    Sequence.block_size = BS
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        eos=(EOS,),
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=BS,
        enable_prefix_caching=True,
    )
    return Scheduler(config)


def make_seq(n, start=0, max_tokens=16, ignore_eos=True):
    return Sequence(list(range(start, start + n)), SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=ignore_eos))


def test_prefill_fifo_and_token_budget():
    s = make_scheduler(max_num_batched_tokens=1024)
    seqs = [make_seq(400, start=i * 1000) for i in range(4)]
    for seq in seqs:
        s.add(seq)
    scheduled, is_prefill = s.schedule()
    assert is_prefill
    assert scheduled == seqs[:2]  # 400 + 400 <= 1024 < 1200
    scheduled, is_prefill = s.schedule()
    assert is_prefill and scheduled == seqs[2:]


def test_prefill_blocked_head_does_not_skip():
    s = make_scheduler(max_num_batched_tokens=1024)
    big, small = make_seq(1000, start=0), make_seq(10, start=5000)
    s.add(make_seq(100, start=9000))
    s.add(big)
    s.add(small)
    scheduled, _ = s.schedule()
    # 严格 FIFO：big 放不下时 small 也不能插队
    assert len(scheduled) == 1


def test_max_num_seqs():
    s = make_scheduler(num_blocks=64, max_num_seqs=3)
    for i in range(5):
        s.add(make_seq(10, start=i * 100))
    scheduled, _ = s.schedule()
    assert len(scheduled) == 3


def test_max_num_seqs_caps_running_across_prefill_steps():
    # 每步 token 预算只够 2 条，并发上限是 3：第二步只能再收 1 条。
    s = make_scheduler(num_blocks=64, max_num_seqs=3, max_num_batched_tokens=100, max_model_len=64)
    for i in range(8):
        s.add(make_seq(40, start=i * 100, max_tokens=1))
    scheduled, is_prefill = s.schedule()
    assert is_prefill and len(scheduled) == 2 and len(s.running) == 2
    scheduled, is_prefill = s.schedule()
    assert is_prefill and len(scheduled) == 1 and len(s.running) == 3
    assert len(s.waiting) == 5


def test_decode_after_prefill_and_finish():
    s = make_scheduler()
    seq = make_seq(10, max_tokens=2)
    s.add(seq)
    scheduled, is_prefill = s.schedule()
    s.postprocess(scheduled, [1])
    scheduled, is_prefill = s.schedule()
    assert not is_prefill and scheduled == [seq]
    s.postprocess(scheduled, [2])
    assert seq.status == SequenceStatus.FINISHED
    assert seq.completion_token_ids == [1, 2]
    assert s.is_finished()
    assert s.block_manager.num_free_blocks == 16


def test_eos_stops_generation():
    s = make_scheduler()
    seq = make_seq(10, max_tokens=100, ignore_eos=False)
    s.add(seq)
    scheduled, _ = s.schedule()
    s.postprocess(scheduled, [EOS])
    assert seq.is_finished


def test_decode_preemption_and_rollback():
    # 4 个块，max_model_len=1024 恰好需要 4 块
    s = make_scheduler(num_blocks=4, max_model_len=1024)
    a, b, c = make_seq(BS, start=0), make_seq(BS, start=10000), make_seq(BS, start=20000)
    for seq in (a, b, c):
        s.add(seq)
    scheduled, is_prefill = s.schedule()
    assert is_prefill and scheduled == [a, b, c]
    s.postprocess(scheduled, [1, 1, 1])
    # 每条序列长度 257 都需要新块，但只剩 1 个空闲块：
    # a 拿走空闲块；b 放不下时抢占队尾的 c，c 释放的块刚好给 b
    scheduled, is_prefill = s.schedule()
    assert not is_prefill
    assert scheduled == [a, b]
    assert s.num_preemptions == 1
    assert list(s.waiting) == [c]
    assert c.status == SequenceStatus.WAITING
    assert c.block_table == []
    # 已生成的 token 保留，恢复时整体 recompute
    assert c.completion_token_ids == [1]
    assert list(s.running) == [a, b]


def test_preempted_seq_resumes():
    s = make_scheduler(num_blocks=4, max_model_len=1024)
    a, b = make_seq(BS, start=0, max_tokens=2), make_seq(BS, start=10000, max_tokens=2)
    s.add(a)
    s.add(b)
    scheduled, _ = s.schedule()
    s.postprocess(scheduled, [1, 1])
    # 人为占满剩余块，迫使 decode 抢占 b
    hog = make_seq(BS * 2, start=50000)
    s.block_manager.allocate(hog)
    scheduled, _ = s.schedule()
    assert scheduled == [a]
    s.postprocess(scheduled, [2])
    assert a.is_finished
    s.block_manager.deallocate(hog)
    scheduled, is_prefill = s.schedule()
    assert is_prefill and scheduled == [b]
    assert len(b) == BS + 1
