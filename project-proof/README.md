# CUDA Reduction Project Proof Pack

## Overview
This proof pack records the real benchmark results of a handwritten CUDA reduction optimization experiment.

## Included Materials
- `docs/experiment-summary.md`: detailed experiment summary for documentation and interview use
- `data/benchmark_results.csv`: structured benchmark record
- `data/experiment_env.csv`: experiment environment and configuration metadata
- `scripts/plot_latency.py`: plotting script for latency comparison
- `scripts/plot_correctness.py`: plotting script for correctness comparison
- `src_refs/key-results.txt`: raw key result snapshot

## Experiment Setup
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- CUDA: `13.0`
- OS: `Ubuntu 24.04.3 LTS (WSL2)`
- Input size: `1 << 24`
- baseline: single-thread GPU reduction (`<<<1,1>>>`)
- v0: block-level shared-memory tree reduction (`block size = 256`)

## Core Results
- CPU: `1.67772e+07`
- baseline GPU: `1.67772e+07`
- baseline Diff: `0`
- baseline latency: `347.757 ms`
- v0 GPU: `1.67772e+07`
- v0 Diff: `0`
- v0 latency: `0.46624 ms`

## Speedup
- v0 vs baseline: `745.86x`
