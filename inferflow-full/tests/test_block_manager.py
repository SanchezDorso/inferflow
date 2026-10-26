from inferflow.engine.block_manager import BlockManager
from inferflow.engine.sequence import Sequence

BS = 256


def make_seq(token_ids):
    Sequence.block_size = BS
    return Sequence(list(token_ids))


def test_allocate_and_free():
    bm = BlockManager(num_blocks=8, block_size=BS)
    seq = make_seq(range(600))  # 3 blocks
    assert bm.can_allocate(seq)
    bm.allocate(seq)
    assert len(seq.block_table) == 3
    assert bm.num_free_blocks == 5
    assert seq.num_cached_tokens == 0
    bm.deallocate(seq)
    assert bm.num_free_blocks == 8
    assert seq.block_table == []


def test_prefix_cache_hit_and_refcount():
    bm = BlockManager(num_blocks=8, block_size=BS)
    shared = list(range(512))
    a = make_seq(shared + [1, 2, 3])
    b = make_seq(shared + [4, 5])
    bm.allocate(a)
    bm.allocate(b)
    assert b.num_cached_tokens == 512
    assert a.block_table[:2] == b.block_table[:2]
    assert a.block_table[2] != b.block_table[2]
    assert bm.blocks[a.block_table[0]].ref_count == 2
    assert bm.num_free_blocks == 8 - 4

    bm.deallocate(a)
    assert bm.blocks[b.block_table[0]].ref_count == 1
    bm.deallocate(b)
    assert bm.num_free_blocks == 8


def test_prefix_cache_survives_free():
    bm = BlockManager(num_blocks=8, block_size=BS)
    a = make_seq(list(range(300)))
    bm.allocate(a)
    first_block = a.block_table[0]
    bm.deallocate(a)
    b = make_seq(list(range(300)))
    bm.allocate(b)
    assert b.num_cached_tokens == 256
    assert b.block_table[0] == first_block


def test_full_prompt_hit_keeps_last_block():
    bm = BlockManager(num_blocks=8, block_size=BS)
    a = make_seq(list(range(512)))
    bm.allocate(a)
    b = make_seq(list(range(512)))
    bm.allocate(b)
    # 最后一块必须重新计算，否则 prefill 没有 token 可产出 logits
    assert b.num_cached_tokens == 256
    assert len(b) - b.num_cached_tokens > 0


def test_same_tokens_different_prefix_no_false_hit():
    bm = BlockManager(num_blocks=3, block_size=BS)
    p1, p2, t = list(range(256)), list(range(5000, 5256)), list(range(1000, 1256))
    a = make_seq(p1 + t)
    bm.allocate(a)
    a_block1 = a.block_table[1]
    bm.deallocate(a)
    # c 的第二块（内容同为 t，但前缀不同）落在 a 原来的第二块上，
    # hash_to_block_id 里仍残留 hash(p1, t) -> 该块 的陈旧映射
    c = make_seq(p2 + t)
    bm.allocate(c)
    assert c.block_table[1] == a_block1
    assert bm.blocks[a_block1].token_ids == t
    bm.deallocate(c)
    b = make_seq(p1 + t + [9])
    bm.allocate(b)
    assert b.num_cached_tokens == 256


def test_can_allocate_ignores_shared_prefix_blocks():
    bm = BlockManager(num_blocks=4, block_size=BS)
    shared = list(range(512))
    a = make_seq(shared + [1, 2, 3])
    bm.allocate(a)
    assert bm.num_free_blocks == 1
    b = make_seq(shared + [4, 5])
    needed, cached = bm.preview(b)
    assert cached == 512
    assert needed == 1
    assert bm.can_allocate(b)
    bm.allocate(b)
    assert b.num_cached_tokens == cached
    assert bm.num_free_blocks == 0


def test_prefix_caching_disabled():
    bm = BlockManager(num_blocks=8, block_size=BS, enable_prefix_caching=False)
    a = make_seq(list(range(600)))
    b = make_seq(list(range(600)))
    bm.allocate(a)
    bm.allocate(b)
    assert b.num_cached_tokens == 0
    assert set(a.block_table).isdisjoint(b.block_table)


def test_may_append_allocates_on_block_boundary():
    bm = BlockManager(num_blocks=4, block_size=BS)
    seq = make_seq(list(range(BS)))
    bm.allocate(seq)
    assert len(seq.block_table) == 1
    seq.append_token(1)  # len = 257
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 2
    for _ in range(BS - 1):
        seq.append_token(2)
        bm.may_append(seq)
    # 第二块写满后应登记 hash，可被后续请求复用
    assert len(seq) == 2 * BS
    assert bm.blocks[seq.block_table[1]].hash != -1
