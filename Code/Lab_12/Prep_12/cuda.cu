#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_select_insert(CudaImg img_big, CudaImg img_small, int2 pos, bool select)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_small.m_size.x || x + pos.x >= img_big.m_size.x) return;
    if (y >= img_small.m_size.y || y + pos.x >= img_big.m_size.y) return;

    if (select)
        img_small.at3(y, x) = img_big.at3(pos.y + y, pos.x + x);
    else
        img_big.at3(pos.y + y, pos.x + x) = img_small.at3(y, x);
}

__global__ void kernel_rotate_90(CudaImg img_orig, CudaImg img_rotated, bool clockwise)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_orig.m_size.x) return;
    if (y >= img_orig.m_size.y) return;

    int newX, newY;

    if (clockwise)
    {
        newX = img_orig.m_size.y - y - 1;
        newY = x;
    }
    else
    {
        newX = y;
        newY = img_orig.m_size.x - x - 1;
    }

    if (newX >= img_rotated.m_size.x) return;
    if (newY >= img_rotated.m_size.y) return;

    img_rotated.at3(newY, newX) = img_orig.at3(y, x);
}



void cu_select_insert( CudaImg &img_big, CudaImg &img_small, int2 pos, bool select )
{
    cudaError_t cudaErr;

    int block_size = 16;
    dim3 block_count;
    block_count.x = img_small.m_size.x / block_size + (img_small.m_size.x % block_size ? 1 : 0);
    block_count.y = img_small.m_size.y / block_size + (img_small.m_size.y % block_size ? 1 : 0);

    kernel_select_insert<<<block_count, dim3(block_size, block_size)>>>(img_big, img_small, pos, select);

    if ((cudaErr = cudaGetLastError()) != cudaSuccess)
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );

    cudaDeviceSynchronize();
}

void cu_rotate_90( CudaImg &img_orig, CudaImg &img_rotated, bool clockwise )
{
    cudaError_t cudaErr;

    int block_size = 16;
    dim3 block_count;
    block_count.x = img_orig.m_size.x / block_size + (img_orig.m_size.x % block_size ? 1 : 0);
    block_count.y = img_orig.m_size.y / block_size + (img_orig.m_size.y % block_size ? 1 : 0);

    kernel_rotate_90<<<block_count, dim3(block_size, block_size)>>>(img_orig, img_rotated, clockwise);

    if ((cudaErr = cudaGetLastError()) != cudaSuccess)
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );

    cudaDeviceSynchronize();
}