#include "reduce_common.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main()
{
    const int N = 1 << 24;

    // host
    std::vector<float> h_in(N, 1.0f);

    // device
    float* d_in = nullptr;
    float* d_out = nullptr;

    // allocate GPU memory
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    // h2d
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // cpu compute
    float cpu = 0.0f;
    for (int i = 0; i < N; i++)
    {
        cpu += h_in[i];
    }

    // create cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------------- baseline ----------------
    float baseline_gpu = 0.0f;
    float baseline_ms = 0.0f;

    // warmup
    reduce_baseline(d_in, d_out, N);
    cudaDeviceSynchronize();

    // timing
    cudaEventRecord(start);
    reduce_baseline(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&baseline_ms, start, stop);

    // D2H
    cudaMemcpy(&baseline_gpu, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // ---------------- v0 ----------------
    float v0_gpu = 0.0f;
    float v0_ms = 0.0f;

    // warmup
    reduce_v0(d_in, d_out, N);
    cudaDeviceSynchronize();

    // timing
    cudaEventRecord(start);
    reduce_v0(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&v0_ms, start, stop);

    // D2H
    cudaMemcpy(&v0_gpu, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // print
    std::cout << "CPU: " << cpu << std::endl;

    std::cout << "baseline GPU: " << baseline_gpu << std::endl;
    std::cout << "baseline Diff: " << std::fabs(cpu - baseline_gpu) << std::endl;
    std::cout << "[baseline] " << baseline_ms << " ms" << std::endl;

    std::cout << "v0 GPU: " << v0_gpu << std::endl;
    std::cout << "v0 Diff: " << std::fabs(cpu - v0_gpu) << std::endl;
    std::cout << "[v0] " << v0_ms << " ms" << std::endl;

    // free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}