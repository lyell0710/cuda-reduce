# cuda-reduce

手写 CUDA reduction 优化实验项目，用于学习、基准测试和可复现记录。

## 项目目标
- 从 `baseline` 逐步优化到 `v5`
- 对比不同版本的性能与正确性
- 生成结构化数据（CSV）和图表（PNG）用于文档留档

## 当前实现版本
- `baseline`: 单线程 GPU 归约基线
- `v0`: shared-memory tree reduction
- `v1` / `v2`: 逐步优化的 block 内归约策略
- `v3`: 每线程处理两个元素，减少访存轮次
- `v4`: warp 尾归约优化，减少尾部同步开销
- `v5`: 在 `v4` 基础上做 block 内循环展开

## 最新基准快照（baseline ~ v5）
- 输入规模：`N = 1 << 24`
- baseline: `350.609680 ms`
- v0: `0.512106 ms` (`684.64x`)
- v1: `0.613744 ms` (`571.26x`)
- v2: `0.601303 ms` (`583.08x`)
- v3: `0.391285 ms` (`896.05x`)
- v4: `0.391077 ms` (`896.52x`)
- v5: `0.394127 ms` (`889.58x`)
- 正确性：所有版本与 CPU 对齐（`Diff = 0`）

> 说明：基准结果会随设备温度、功耗策略、后台负载出现小幅波动。

## 目录结构
- `src/`: 各版本 kernel 与 benchmark 入口
- `include/`: 公共声明
- `project-proof/data/`: 基准与环境 CSV
- `project-proof/scripts/`: 画图脚本
- `project-proof/docs/`: 图表与实验总结

## 构建与运行
```bash
cmake -S . -B build
cmake --build build -j
./build/reduce_bench
```

## 基准流程说明
- `main.cu` 采用多次迭代计时取均值（当前为 `100` 次）
- 每次运行 `reduce_bench` 会自动覆盖刷新：`project-proof/data/benchmark_results.csv`

## 生成图表
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_latency_line.py
python project-proof/scripts/plot_correctness.py
```

## 相关文档
- 基准数据：`project-proof/data/benchmark_results.csv`
- 实验总结：`project-proof/docs/experiment-summary.md`
