# CUDA Reduction Optimization Record

## Result Summary
- Implementation: handwritten CUDA baseline and v0 reduction kernels
- CPU result: `1.67772e+07`
- baseline GPU result: `1.67772e+07`
- baseline Diff: `0`
- baseline latency: `347.757 ms`
- v0 GPU result: `1.67772e+07`
- v0 Diff: `0`
- v0 latency: `0.46624 ms`

## Correctness
- baseline matches CPU reference: `Diff = 0`
- v0 matches CPU reference: `Diff = 0`
- Both implementations passed correctness validation on the current input

## Baseline vs v0
- baseline latency: `347.757 ms`
- v0 latency: `0.46624 ms`
- v0 preserves numerical correctness while significantly reducing kernel execution time

## Speedup
- Formula: `speedup = baseline_time / optimized_time`
- Computation: `347.757 / 0.46624 = 745.86`
- Speedup: **`745.86x`**

## Experiment Environment
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- GPU memory: `8188 MiB`
- Driver version: `581.57`
- CUDA toolkit: `13.0 (V13.0.88)`
- OS: `Ubuntu 24.04.3 LTS (WSL2, Linux 6.6.87.2-microsoft-standard-WSL2)`
- CPU: `AMD Ryzen 9 7945HX with Radeon Graphics`
- CMake: `3.28.3`
- Host compiler: `g++ 13.3.0`

## Problem Configuration
- Input size: `N = 1 << 24`
- Data type: `float`
- Reduction target: single scalar sum
- baseline launch configuration: `<<<1,1>>>`
- v0 block size: `256`

## Timing Method
- Timing was measured using CUDA events
- A warmup run was executed before each timed run
- Host-to-device transfer was completed before timing
- Device-to-host transfer was performed after timing

## Baseline Definition
- baseline uses a single CUDA block and a single CUDA thread (`<<<1,1>>>`)
- the kernel performs the full accumulation serially on the GPU
- the result is written to device output memory after accumulation completes

## v0 Definition
- v0 uses block-level shared memory reduction
- each kernel pass reduces the input into block-level partial sums
- the host wrapper repeatedly launches the kernel until only one value remains

## Why baseline and v0 differ
- baseline is effectively a serial GPU implementation
- v0 introduces block-level parallelism and shared-memory-based tree reduction
- both implementations produced numerically identical results in the current experiment

## Recommended Experiment Record Fields
- project_name
- experiment_date
- version
- input_size
- cpu_result
- gpu_result
- diff
- latency_ms
- speedup
- correctness_pass
- commit_id
- gpu_model
- cuda_version
- build_command
- notes

## Technical Note
This experiment evaluates CUDA reduction kernel optimization from a baseline implementation to a v0 optimized version. The baseline GPU result matches the CPU reference and records a latency of `347.757 ms`. The v0 implementation also matches the CPU reference with `Diff = 0`, while reducing latency to `0.46624 ms`. Based on measured data, the v0 kernel achieves approximately `745.86x` speedup over the baseline. The result demonstrates that the current reduction optimization direction is effective and has been validated on both correctness and performance.
