#include <iostream>
#include <cuda_runtime.h>

__global__ void cudaHelloKernel() {
    printf("Hello from the CUDA kernel! (thread %d)\n", threadIdx.x);
}

void launchCudaHello() {
    std::cout << "Launching CUDA kernel... with glfw" << std::endl;

    // Launch with 1 block of 4 threads for demonstration
    cudaHelloKernel<<<1, 4>>>();
    
    // Ensure output is flushed 
    cudaDeviceSynchronize();
}
