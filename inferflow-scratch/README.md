# InferFlow（从零实现版）

这是 InferFlow 的**脚手架**：环境脚本、目录结构、接口签名、注释提示和逐步测试都已就绪，核心逻辑以 `TODO(StepN)` 留空，按下面的路线从零实现。完整参考实现在同级目录 `../inferflow-full`，卡住时可以对照。

最终要实现的能力：

- 张量并行（TP）多进程推理：共享内存 RPC + NCCL
- 连续批处理，最多 512 条并发序列
- 分页 KV Cache + FlashAttention，支持 4096 上下文
- 两阶段调度：Prefill FIFO、Decode 抢占 + 回滚
- 基于哈希的前缀缓存复用 + 引用计数
- CUDA Graph 加速 decode

## Step 0：部署环境（已就绪）

Step 1 ~ 8 **不需要 GPU**：`device="cpu"` 调试路径会用 `layers/attention.py` 里已提供的 SDPA 参考注意力，测试使用随机初始化的迷你 Qwen3，无需下载模型。Step 9 需要 GPU。

```bash
# GPU 机器（完整环境：torch cu126 + flash-attn 预编译 wheel + 本项目 editable 安装）
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/download_model.sh Qwen/Qwen3-0.6B ~/huggingface/Qwen3-0.6B   # 国内: HF_ENDPOINT=https://hf-mirror.com

# 仅 CPU 开发（Step 1 ~ 8）
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.51" safetensors xxhash numpy tqdm pytest
```

检查某一步是否完成：

```bash
bash scripts/check_step.sh 1      # 1 ~ 9，或 all
```

## 实现路线

每一步只需要改标注了对应 `TODO(StepN)` 的位置（`grep -rn "TODO(Step3)" inferflow` 可快速定位）。

| 步骤 | 内容 | 涉及文件 | 验证 |
| --- | --- | --- | --- |
| 1 | 序列与 KV 块分配（无前缀缓存） | `engine/sequence.py`、`engine/block_manager.py` | `check_step.sh 1` |
| 2 | 调度器：Prefill FIFO + Decode + 结束判定；`can_append / may_append` | `engine/scheduler.py`、`engine/block_manager.py` | `check_step.sh 2` |
| 3 | 张量并行层：列并行 / 行并行 / QKV / MergedColumn / 词表并行 Embedding & LMHead | `layers/linear.py`、`layers/embed_head.py` | `check_step.sh 3` |
| 4 | Qwen3 模型：RMSNorm、RoPE、SiLU、注意力前向、权重加载 | `layers/layernorm.py`、`layers/rotary_embedding.py`、`layers/activation.py`、`layers/attention.py`（`Attention.forward`）、`models/qwen3.py`、`utils/loader.py` | `check_step.sh 4` |
| 5 | 分页 KV 写入 + ModelRunner（KV 显存规划、prefill/decode 输入准备）+ 采样 + 引擎 step | `layers/attention.py`（`store_kvcache`）、`layers/sampler.py`、`engine/model_runner.py`、`engine/llm_engine.py` | `check_step.sh 5` |
| 6 | 前缀缓存：链式哈希匹配、引用计数、命中时 prefill 从分页 cache 读 K/V | `engine/block_manager.py`（`allocate`）、`engine/model_runner.py`（`prepare_prefill`）、`layers/attention.py` | `check_step.sh 6` |
| 7 | 抢占与回滚：decode 显存不足时抢占队尾、回滚为 WAITING、recompute 恢复 | `engine/scheduler.py` | `check_step.sh 7` |
| 8 | 多进程张量并行：共享内存 RPC、KV 块数跨 rank 取 min、`Sequence` 精简序列化 | `engine/model_runner.py`、`engine/sequence.py` | `check_step.sh 8` |
| 9 | GPU：FlashAttention paged 接口、Triton `store_kvcache` kernel、CUDA Graph、跑 benchmark | `layers/attention.py`、`engine/model_runner.py` | `check_step.sh 9`、`python bench.py` |

### 各步要点

**Step 1：序列与块**  
`Sequence` 是请求在引擎内的表示；`block_table[i]` 把逻辑块 i 映射到物理块。`BlockManager` 用 `free_block_ids`（deque）和 `used_block_ids`（set）管理物理块，从队头分配、释放时放回队尾。

**Step 2：调度器**  
Prefill 严格 FIFO：队头放不下就停，不允许后面的短请求插队。只要有 prefill 就先做 prefill，否则做一步 decode。decode 前 `may_append` 为新 token 预留槽位（跨块边界时分配新块，块写满时登记 hash）。

**Step 3：张量并行**  
Column（按输出维切分，无通信）→ Row（按输入维切分，all-reduce）是 MLP / Attention 的标准组合，每层只需要 2 次 all-reduce。`weight_loader` 负责从完整 checkpoint 张量中切出本 rank 的分片。测试用 2 个 gloo 进程在 CPU 上验证与单卡等价。

**Step 4：模型**  
Qwen3 = Embedding → N × (RMSNorm → Attention(QKV → q/k norm → RoPE → attn → O) → RMSNorm → MLP) → RMSNorm → LMHead。多条序列拼成一维 `[total_tokens, hidden]`，用 `cu_seqlens` 标出边界（varlen），没有 padding。

**Step 5：分页 KV + 引擎**  
`slot_mapping` 告诉注意力层每个新 token 的 K/V 写到 cache 的哪个位置；decode 时 `block_tables + context_lens` 告诉注意力去哪些块读历史 KV。KV cache 容量 = 可用显存减去 warmup 测得的激活峰值，再除以每块字节数。

**Step 6：前缀缓存**  
`h_i = xxh64(h_{i-1} || tokens_i)`，hash 相同说明整段前缀相同。命中的块共享（`ref_count += 1`），只计算未命中的部分：此时 Q 比 K 短，prefill 注意力要从分页 cache 读完整的 K/V。有三个坑，测试都会覆盖到：
- 最后一块必须重算
- 释放后的块仍然可以被命中
- 陈旧映射要同时校验 hash 和 token_ids

**Step 7：抢占**  
某条序列需要新块但空闲块不够时，从 running 队尾抢占：释放全部块、状态回滚为 WAITING、插回 waiting 队首。已生成的 token 保留，下次与 prompt 一起重新 prefill。测试通过把 `num_kvcache_blocks` 设得很小来强制触发抢占。

**Step 8：多进程**  
rank 0 在主进程里调度和采样，rank 1..N-1 用 `mp.get_context("spawn")` 拉起，停在 `loop()` 里等待指令。rank 0 的 `call(method, *args)` 把参数 pickle 进共享内存，再用 Event 唤醒各 worker，所有 rank 同步执行同一个方法。

**Step 9：GPU 优化**  
- `store_kvcache` 从布尔索引改成 Triton kernel，这样可以被 CUDA Graph 录制。
- decode 按 batch size 分档录制 CUDA Graph，replay 前把输入拷进静态 buffer，padding 位置的 `slot_mapping` 填 -1。
- 最后用 `bench.py` 的 `random` / `prefix` / `long` 三个场景测吞吐。

## 已提供的部分

- `config.py`、`sampling_params.py`、`llm.py`、`utils/context.py`：配置与接口
- `layers/attention.py` 中的 `reference_attention`：CPU 参考注意力（语义与 flash_attn 调用一致）
- `engine/llm_engine.py`：进程拉起、`generate()` 主循环（`step()` 待实现）
- `engine/model_runner.py`：初始化流程、`call/loop`、`run()` 骨架
- `scripts/`：环境安装、模型下载、环境自检、分步测试
- `tests/`：全部测试（与完整版相同）
- `example.py`、`bench.py`
