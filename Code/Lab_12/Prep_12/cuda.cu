#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_select_insert(CudaImg big_img, CudaImg small_img, int2 pos, bool select)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= small_img.m_size.x || x + pos.x >= big_img.m_size.x) return;
    if (y >= small_img.m_size.y || y + pos.x >= big_img.m_size.y) return;

    if (select)
        small_img.m_p_uchar3[y * small_img.m_size.x + x] = big_img.m_p_uchar3[(pos.y + y) * big_img.m_size.x + (pos.x + x)];
    else
        big_img.m_p_uchar3[(pos.y + y) * big_img.m_size.x + (pos.x + x)] = small_img.m_p_uchar3[y * small_img.m_size.x + x];
}

__global__ void kernel_rotate_90(CudaImg original, CudaImg rotated, bool clockwise)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= original.m_size.x) return;
    if (y >= original.m_size.y) return;

    int newX, newY;

    if (clockwise)
    {
        newX = original.m_size.y - y - 1;
        newY = x;
    }
    else
    {
        newX = y;
        newY = original.m_size.x - x - 1;
    }

    if (newX >= rotated.m_size.x) return;
    if (newY >= rotated.m_size.y) return;

    rotated.m_p_uchar3[newY * rotated.m_size.x + newX] = original.m_p_uchar3[y * original.m_size.x + x];
}



void cu_select_insert( CudaImg &t_big_rgb, CudaImg &t_small_rgb, int2 t_pos, bool select )
{
    cudaError_t cudaErr;

    int blockSize = 16;
    dim3 blockCount;
    blockCount.x = t_small_rgb.m_size.x / blockSize + (t_small_rgb.m_size.x % blockSize ? 1 : 0);
    blockCount.y = t_small_rgb.m_size.y / blockSize + (t_small_rgb.m_size.y % blockSize ? 1 : 0);

    kernel_select_insert<<<blockCount, dim3(blockSize, blockSize)>>>(t_big_rgb, t_small_rgb, t_pos, select);

    if ((cudaErr = cudaGetLastError()) != cudaSuccess)
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );

    cudaDeviceSynchronize();
}

void cu_rotate_90( CudaImg &t_in_rgb, CudaImg &t_rot_rgb, bool clockwise )
{
    cudaError_t cudaErr;

    int blockSize = 16;
    dim3 blockCount;
    blockCount.x = t_in_rgb.m_size.x / blockSize + (t_in_rgb.m_size.x % blockSize ? 1 : 0);
    blockCount.y = t_in_rgb.m_size.y / blockSize + (t_in_rgb.m_size.y % blockSize ? 1 : 0);

    kernel_rotate_90<<<blockCount, dim3(blockSize, blockSize)>>>(t_in_rgb, t_rot_rgb, clockwise);

    if ((cudaErr = cudaGetLastError()) != cudaSuccess)
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );

    cudaDeviceSynchronize();
}