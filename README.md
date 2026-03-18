# cuda-reduce

A handwritten CUDA reduction project for learning, benchmarking, and documenting kernel optimization.

## what this project shows
- implementation of a baseline CUDA reduction kernel
- implementation of a v0 shared-memory tree reduction kernel
- correctness verification against a CPU reference
- reproducible benchmark records and proof materials

## current result
- baseline: `347.757 ms`
- v0: `0.46624 ms`
- speedup: `745.86x`
- correctness: both GPU versions matched the CPU reference (`Diff = 0`)

## repository layout
- `src/`: kernel implementations and benchmark driver
- `include/`: shared interfaces
- `project-proof/`: experiment notes, benchmark data, and figures

## build
```bash
cmake -S . -B build
cmake --build build -j
./build/reduce_bench
