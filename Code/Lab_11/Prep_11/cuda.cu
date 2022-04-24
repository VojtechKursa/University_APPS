#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

__global__ void kernel_add_arrays(int* arrIn1, int* arrIn2, int* arrOut, int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length) return;

    arrOut[i] = arrIn1[i] + arrIn2[i];
}


void cu_add_arrays(int* arrIn1, int* arrIn2, int* arrOut, int length)
{
    int threadsPerBlock = 128;
    int blockCount = length / threadsPerBlock + (length % threadsPerBlock ? 1 : 0);

    int lengthInBytes = sizeof(int) * length;
    int *arrIn1GPU, *arrIn2GPU, *arrOutGPU;

    cudaMalloc(&arrIn1GPU, lengthInBytes);
    cudaMalloc(&arrIn2GPU, lengthInBytes);
    cudaMalloc(&arrOutGPU, lengthInBytes);

    cudaMemcpy(arrIn1GPU, arrIn1, lengthInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(arrIn2GPU, arrIn2, lengthInBytes, cudaMemcpyHostToDevice);

    kernel_add_arrays<<<blockCount, threadsPerBlock>>>(arrIn1GPU, arrIn2GPU, arrOutGPU, length);

    cudaMemcpy(arrOut, arrOutGPU, lengthInBytes, cudaMemcpyDeviceToHost);

    cudaFree(arrIn1GPU);
    cudaFree(arrIn2GPU);
    cudaFree(arrOutGPU);
}