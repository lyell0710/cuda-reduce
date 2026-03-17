#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "reduce_common.h"


int main(){
   const int N = 1 << 24;

   //host
   std::vector<float> h_in(N,1.0f);

   //device
   float* d_in= nullptr;
   float* d_out = nullptr;

   //allocate GPU memory
   cudaMalloc(&d_in, N*sizeof(float));
   cudaMalloc(&d_out, sizeof(float));


   //h2d
   cudaMemcpy(d_in ,h_in.data(),N * sizeof(float), cudaMemcpyHostToDevice);

   //warmup
   reduce_baseline(d_in, d_out,N);
   cudaDeviceSynchronize();

   //creat cuda event 

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);


   // timing the kernel
   cudaEventRecord(start);
   reduce_baseline(d_in,d_out,N);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   float ms = 0.0f;
   cudaEventElapsedTime(&ms,start, stop);

   //D2H
   float gpu= 0.0f;
   cudaMemcpy(&gpu,d_out,sizeof(float),cudaMemcpyDeviceToHost);

   // cpu compute
   float cpu = 0.0f;
    for (int i = 0; i < N; i++) {
        cpu += h_in[i];
    }

    //打印结果和耗时
    std::cout << "CPU: " << cpu << std::endl;
    std::cout << "GPU: " << gpu << std::endl;
    std::cout << "Diff: " << std::fabs(cpu - gpu) << std::endl;
    std::cout << "[baseline] " << ms << " ms" << std::endl;
   

    // 11. 释放 GPU 显存
    cudaFree(d_in);
    cudaFree(d_out);
   
    return 0;

}

