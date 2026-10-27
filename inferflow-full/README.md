# InferFlow

轻量级大语言模型推理框架，基于 PyTorch 从零实现，约 1500 行核心代码。支持：

- **张量并行（TP）多进程推理**：rank 0 负责调度与采样，其余 rank 在子进程中执行；控制面通过共享内存 + Event 做 RPC，数据面通过 NCCL all-reduce / gather 通信
- **连续批处理（Continuous Batching）**：每一步重新调度，已完成的序列立即释放资源，新请求随时加入；默认最多 512 条并发序列
- **分页 KV Cache（PagedAttention）**：KV 按 16 token 为一块管理，prefill / decode 通过 `block_table` 调用 FlashAttention 3
- **两阶段调度 + 抢占回滚**：Prefill 严格 FIFO；Decode 显存不足时从队尾抢占，释放 KV 块、状态回滚为 WAITING，之后以 recompute 方式恢复
- **前缀缓存（Prefix Caching）**：链式哈希匹配已有 KV 块，引用计数共享，空闲块保留内容当作 LRU 缓存
- **CUDA Graph**：decode 阶段按 batch size 分档录制，减少 kernel launch 开销
- 模型：Qwen3 系列（已验证 Qwen3-0.6B），上下文最长 4096（可配置）

## 目录结构

```
inferflow/
├── llm.py                  # 对外接口 LLM（= LLMEngine）
├── config.py               # 全局配置
├── sampling_params.py      # 采样参数（temperature=0 为贪心）
├── engine/
│   ├── sequence.py         # 请求序列：token、状态、block_table
│   ├── block_manager.py    # KV 块分配 / 释放 / 前缀哈希 / 引用计数
│   ├── scheduler.py        # Prefill FIFO + Decode 抢占调度
│   ├── model_runner.py     # 单 rank 执行器：TP 进程组、共享内存 RPC、KV 显存规划、输入准备、CUDA Graph
│   └── llm_engine.py       # 引擎主循环：拉起 TP worker、step()、generate()
├── layers/
│   ├── linear.py           # Column / Row / QKV / MergedColumn 并行线性层
│   ├── embed_head.py       # 词表并行 Embedding / LMHead
│   ├── attention.py        # Triton 写 KV cache + FlashAttention 3（paged）
│   ├── rotary_embedding.py # RoPE
│   ├── layernorm.py        # RMSNorm（融合 residual add）
│   ├── activation.py       # SiLU * gate
│   └── sampler.py          # 温度采样（Gumbel-max）/ 贪心
├── models/qwen3.py         # Qwen3 模型定义
└── utils/
    ├── context.py          # 前向注意力元数据（cu_seqlens / slot_mapping / block_tables ...）
    └── loader.py           # safetensors 权重加载（按 TP rank 切分）
scripts/
├── setup_env.sh            # GPU 机器一键安装环境（torch + flash-attn-3 预编译 wheel）
├── download_model.sh       # 下载模型
└── check_env.py            # 环境自检
tests/                      # 单测 + 端到端对比测试
example.py / bench.py       # 示例与吞吐基准
```

## 部署

需要 Linux + NVIDIA GPU（Ampere 及以上，FlashAttention 3），Python 3.10~3.12。

```bash
# 1. 安装环境（默认创建 ./.venv；torch 2.9.1 + cu126 + flash-attn-3 3.0.1）
bash scripts/setup_env.sh
#   国内网络可加：PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   驱动较旧可换：TORCH_CUDA=cu118 或 cu124
source .venv/bin/activate

# 2. 下载模型（国内：HF_ENDPOINT=https://hf-mirror.com）
bash scripts/download_model.sh Qwen/Qwen3-0.6B ~/huggingface/Qwen3-0.6B

# 3. 运行
python example.py --model ~/huggingface/Qwen3-0.6B
python example.py --model ~/huggingface/Qwen3-0.6B --tp 2
python bench.py --model ~/huggingface/Qwen3-0.6B
python bench.py --scenario prefix
python bench.py --scenario long
```

## 使用

```python
from inferflow import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B", tensor_parallel_size=1, max_model_len=4096)
outputs = llm.generate(["你好，介绍一下你自己"], SamplingParams(temperature=0.6, max_tokens=256))
print(outputs[0]["text"])
```

主要配置（`LLM(model, **kwargs)`）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `tensor_parallel_size` | 1 | TP 卡数 |
| `max_num_seqs` | 512 | 同时运行的最大序列数 |
| `max_num_batched_tokens` | 16384 | 单次 prefill 的 token 上限 |
| `max_model_len` | 4096 | 单序列最大长度 |
| `gpu_memory_utilization` | 0.9 | KV cache 可用显存比例 |
| `kvcache_block_size` | 16 | KV 块大小，也是前缀缓存的命中粒度 |
| `num_kvcache_blocks` | -1 | >0 时限制 KV 块数，便于复现抢占 |
| `enable_prefix_caching` | True | 前缀缓存 |
| `enforce_eager` | False | 关闭 CUDA Graph |

## 测试

```bash
pytest tests/test_block_manager.py tests/test_scheduler.py   # 纯逻辑，秒级
pytest tests/test_tp_layers.py                               # 2 进程 gloo 验证 TP 层等价性（CPU）
pytest tests/test_model_forward.py                           # 单次 prefill 前向 logits 对比 HF（CPU）
pytest tests/test_engine_cpu.py                              # 迷你随机 Qwen3，端到端对比 HF（CPU）
MODEL_PATH=~/huggingface/Qwen3-0.6B pytest tests/test_qwen3_gpu.py   # GPU，对比 HF 真实模型
```

`device="cpu"` 是仅供调试的路径：注意力退化为逐序列的 SDPA 参考实现，使用 gloo 通信，不启用 CUDA Graph，用于在没有 GPU 的机器上验证调度、分页 KV、前缀缓存、抢占和 TP 的正确性。

## 设计要点

### 一次 step 的流程

```
Scheduler.schedule()          -> (seqs, is_prefill)
ModelRunner.call("run", ...)  -> rank0 写共享内存，唤醒所有 worker
  prepare_prefill / decode    -> input_ids, positions, slot_mapping, block_tables, cu_seqlens
  model forward               -> 每层：QKV -> RoPE -> store_kvcache -> FlashAttention -> O proj (all-reduce)
  LMHead (gather 到 rank0)     -> Sampler
Scheduler.postprocess()       -> 追加 token，判断结束，释放 KV 块
```

### KV Cache 显存规划

启动时先用最大 prefill 批做一次 warmup 前向，拿到激活峰值显存，再按
`total * gpu_memory_utilization - 已用 - 峰值激活` 计算能放下多少个 KV 块。各 rank 取最小值，保证 block id 在所有 rank 上都合法。

### 前缀缓存

每个满块的 hash = `xxh64(前一块 hash || 本块 token_ids)`，因此 hash 相同即前缀完全相同。命中时增加引用计数，直接复用物理块；块释放后仍保留 hash 与内容，直到空闲链表把它重新分配出去。匹配时同时校验 hash 与 token_ids，避免陈旧映射造成误命中。prompt 的最后一块总是重新计算，保证 prefill 至少有一个 token 产生 logits。

### 抢占与回滚

decode 时若某序列需要新块而空闲块不足，就从 running 队尾抢占：释放其全部 KV 块，状态改回 WAITING 并插回 waiting 队首。它已生成的 token 会保留，下一轮与 prompt 一起重新 prefill。结合前缀缓存，这些被释放的块若尚未被覆盖，重算时还能部分命中。
