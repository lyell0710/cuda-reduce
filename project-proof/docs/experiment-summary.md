# CUDA Reduction 实验总结（baseline ~ v5）

## 1) 实验目标
验证 CUDA reduction 从 `baseline` 到 `v5` 的优化收益，比较各版本在同一输入规模下的正确性与延迟表现。

## 2) 核心结论
- 所有版本（`baseline`、`v0`、`v1`、`v2`、`v3`、`v4`、`v5`）均与 CPU 结果一致，`Diff = 0`。
- 当前最优版本为 `v4`：`0.360384 ms`，相对 `baseline`（`350.616 ms`）达到 **`972.90x`** 加速。
- `v5` 延迟为 `0.365568 ms`，与 `v4` 接近，但本次实测略慢于 `v4`。

## 3) 结果总表
| version | cpu_result | gpu_result | diff | latency_ms | speedup_vs_baseline | correctness_pass |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1.67772e+07 | 1.67772e+07 | 0 | 350.616000 | 1.00x | true |
| v0 | 1.67772e+07 | 1.67772e+07 | 0 | 0.469760 | 746.37x | true |
| v1 | 1.67772e+07 | 1.67772e+07 | 0 | 0.467968 | 749.23x | true |
| v2 | 1.67772e+07 | 1.67772e+07 | 0 | 0.456000 | 768.89x | true |
| v3 | 1.67772e+07 | 1.67772e+07 | 0 | 0.378624 | 926.03x | true |
| v4 | 1.67772e+07 | 1.67772e+07 | 0 | 0.360384 | 972.90x | true |
| v5 | 1.67772e+07 | 1.67772e+07 | 0 | 0.365568 | 959.10x | true |

## 4) 环境与配置
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`（`8188 MiB`）
- Driver: `581.57`
- CUDA: `13.0`
- OS: `Ubuntu 24.04.3 LTS (WSL2, Linux 6.6.87.2-microsoft-standard-WSL2)`
- CPU: `AMD Ryzen 9 7945HX with Radeon Graphics`
- CMake: `3.28.3`
- Compiler: `g++ 13.3.0`
- 输入规模: `N = 1 << 24`
- baseline 启动配置: `<<<1,1>>>`
- 计时方式: `CUDA events`，含 warmup

## 5) 图表文件
- `project-proof/docs/figures/latency_comparison.png`
- `project-proof/docs/figures/latency_comparison_log.png`
- `project-proof/docs/figures/latency_comparison_line.png`
- `project-proof/docs/figures/correctness_check.png`

## 6) 简要分析
- `baseline` 使用近似串行的 GPU 归约方式，延迟远高于优化版本。
- `v0`~`v2` 带来稳定提升，但幅度相对接近（约 `746x` 到 `769x`）。
- `v3`~`v5` 进一步拉高性能，其中 `v4` 达到本次峰值（`972.90x`），`v5` 与其接近但略有回退。
