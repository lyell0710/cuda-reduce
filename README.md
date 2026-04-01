# cuda-reduce

A handwritten CUDA reduction project for learning, benchmarking, and documenting kernel optimization.

## what this project shows
- step-by-step CUDA reduction optimization from `baseline` to `v5`
- multiple kernel strategies (`baseline`, `v0`, `v1`, `v2`, `v3`, `v4`, `v5`)
- correctness verification against a CPU reference on each version
- reproducible benchmark records, plots, and summary docs in `project-proof/`

## latest benchmark snapshot
- input size: `N = 1 << 24`
- baseline: `350.616 ms`
- v0: `0.469760 ms` (`746.37x`)
- v1: `0.467968 ms` (`749.23x`)
- v2: `0.456000 ms` (`768.89x`)
- v3: `0.378624 ms` (`926.03x`)
- v4: `0.360384 ms` (`972.90x`)
- v5: `0.365568 ms` (`959.10x`)
- correctness: all GPU versions matched CPU (`Diff = 0`)

## repository layout
- `src/`: kernel implementations and benchmark driver
- `include/`: shared interfaces
- `project-proof/data/`: benchmark/environment CSV records
- `project-proof/scripts/`: plotting scripts
- `project-proof/docs/`: figures and experiment summary

## build
```bash
cmake -S . -B build
cmake --build build -j
./build/reduce_bench
```

## regenerate proof figures
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_latency_line.py
python project-proof/scripts/plot_correctness.py
```

## notes
- benchmark data source: `project-proof/data/benchmark_results.csv`
- full write-up: `project-proof/docs/experiment-summary.md`
