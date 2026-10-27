# Q/K RMSNorm + RoPE 融合 kernel

机器：RTX 4090 D。模型：Qwen3-0.6B，28 层，Q 16 head，K 8 head，head_dim 128，bfloat16。块大小 16。

融合前每层是两次 RMSNorm、一次 RoPE，prefill 再把 K 单独写入页。融合后一个 Triton kernel 在寄存器里做完按 head 的 RMSNorm 和 rotate-half RoPE，Q 只写出一次；prefill 的 K 按 `slot_mapping` 直接写入页，`slot == -1` 跳过。`INFERFLOW_FUSED_QK=0` 走原来的拆开实现。

和拆开的 RMSNorm + RoPE 比，最大绝对误差 0.0078125，约 1 个 bf16 ulp。80 token 对齐的共享前缀第二次请求命中 80 token。

## 算子

一次计时覆盖 28 层。各先热身 10 次，再测 30 次取平均。

| 形状 | 融合前 | 融合后 | 加速 |
| --- | ---: | ---: | ---: |
| prefill，4096 token | 15.746 ms | 2.644 ms | 5.96× |
| decode，batch 32 | 23.648 ms | 1.611 ms | 14.68× |

拆开的实现每次都从 Python 启动编译后的小 kernel，短 batch 上启动开销占了大头。下面的端到端数字把 matmul 和注意力算进去。

## 端到端

同一进程里先关融合、再开融合。每组先热身一次，再跑 5 次，取中位数。前缀缓存关闭，temperature=0，ignore_eos。显存占用 0.45，max_model_len=1024，最多 16 条并发。

16 条请求，prompt 512，生成 1 token：

| | 中位数 | 5 次（ms） |
| --- | ---: | --- |
| 融合前 | 76.85 ms | 76.16, 76.57, 76.85, 78.65, 78.66 |
| 融合后 | 75.33 ms | 75.28, 75.30, 75.33, 75.49, 78.37 |

少 1.52 ms，约 2.0%。这一段主要是 prefill，时间在 matmul 和注意力上。

16 条请求，prompt 64，生成 128 token（输出 2048 token）：

| | 中位数 | 输出吞吐 | 5 次（ms） |
| --- | ---: | ---: | --- |
| 融合前 | 492.40 ms | 4159 tok/s | 480.89, 481.86, 492.40, 492.82, 507.15 |
| 融合后 | 467.23 ms | 4383 tok/s | 465.67, 466.65, 467.23, 468.07, 469.58 |

少 25.17 ms，约 5.1%。输出吞吐从 4159 tok/s 到 4383 tok/s。
