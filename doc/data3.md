# residual add + RMSNorm、SiLU and Mul

机器：RTX 4090 D。模型：Qwen3-0.6B，28 层，hidden 1024，intermediate 3072，bfloat16。每层两次 RMSNorm（input 和 post-attention）、一次 SiLU，28 层走同一份代码。

这两处原来各自在一个 `torch.compile` 区域里：residual 相加和 RMSNorm 已经写在一起，SiLU 和 mul 也已经写在一起。换成 Triton 之后，相加、方差、缩放留在寄存器里，SiLU 和乘法也只写出一次。`INFERFLOW_TRITON_NORM_ACT=0` 走原来的 compile 实现。

随机 bfloat16 输入上，RMSNorm、带 residual 的 RMSNorm、SiLU and Mul 和 compile 版本的最大绝对误差都是 0。

## 算子

一次计时覆盖一整次前向里这些调用：RMSNorm 56 次（28 层 × 2），SiLU 28 次。各先热身 10 次，再测 30 次取平均。

| 形状 | 算子 | 改前 | 改后 | 加速 |
| --- | --- | ---: | ---: | ---: |
| prefill，4096 token | RMSNorm × 56 | 6.833 ms | 2.257 ms | 3.03× |
| prefill，4096 token | SiLU × 28 | 1.883 ms | 0.797 ms | 2.36× |
| decode，batch 32 | RMSNorm × 56 | 7.524 ms | 2.307 ms | 3.26× |
| decode，batch 32 | SiLU × 28 | 2.142 ms | 0.796 ms | 2.69× |

compile 版本每次仍从 Python 启动。短 batch 上两边耗时几乎不随 token 数变，省下的主要是启动和那一次中间写回。下面的端到端数字把 matmul 和注意力算进去。

## 端到端

同一进程里先关 Triton、再打开。每组先热身一次，再跑 5 次，取中位数。temperature=0，ignore_eos，前缀缓存关闭。显存占用 0.45，max_model_len=1024，最多 16 条并发。

16 条请求，prompt 512，生成 1 token：

| | 中位数 | 5 次（ms） |
| --- | ---: | --- |
| 改前 | 76.28 ms | 75.67, 75.95, 76.28, 78.06, 78.14 |
| 改后 | 73.12 ms | 73.01, 73.02, 73.12, 73.22, 73.45 |

少 3.16 ms，约 4.1%。这一段主要是 prefill。

16 条请求，prompt 64，生成 128 token（输出 2048 token）：

| | 中位数 | 输出吞吐 | 5 次（ms） |
| --- | ---: | ---: | --- |
| 改前 | 452.54 ms | 4526 tok/s | 440.60, 441.90, 452.54, 473.47, 473.54 |
| 改后 | 440.10 ms | 4654 tok/s | 437.66, 438.60, 440.10, 440.22, 441.98 |

少 12.44 ms，约 2.7%。输出吞吐从 4526 tok/s 到 4654 tok/s。
