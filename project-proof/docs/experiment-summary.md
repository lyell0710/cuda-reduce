# CUDA Reduction 实验总结（baseline ~ v4）

## 1) 实验目标
验证 CUDA reduction 从 `baseline` 到 `v4` 的优化收益，比较各版本在同一输入规模下的正确性与延迟表现。

## 2) 核心结论
- 所有版本（`baseline`、`v0`、`v1`、`v2`、`v3`、`v4`）均与 CPU 结果一致，`Diff = 0`。
- 优化后性能显著提升：`v4` 延迟为 `0.376800 ms`，相对 `baseline`（`350.859 ms`）达到 **`931.15x`** 加速。
- `v3` 与 `v4` 表现非常接近，`v4` 略优于 `v3`（`0.376800 ms` vs `0.376832 ms`）。

## 3) 结果总表
| version | cpu_result | gpu_result | diff | latency_ms | speedup_vs_baseline | correctness_pass |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1.67772e+07 | 1.67772e+07 | 0 | 350.859000 | 1.00x | true |
| v0 | 1.67772e+07 | 1.67772e+07 | 0 | 0.481216 | 729.11x | true |
| v1 | 1.67772e+07 | 1.67772e+07 | 0 | 0.478624 | 733.06x | true |
| v2 | 1.67772e+07 | 1.67772e+07 | 0 | 0.475488 | 737.89x | true |
| v3 | 1.67772e+07 | 1.67772e+07 | 0 | 0.376832 | 931.08x | true |
| v4 | 1.67772e+07 | 1.67772e+07 | 0 | 0.376800 | 931.15x | true |

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
- `v0`~`v2` 带来稳定提升，但幅度相对接近（约 `729x` 到 `738x`）。
- `v3/v4` 是本轮最有效的优化阶段，速度提升到约 `931x`，并保持与 CPU 数值一致。
