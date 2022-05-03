#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>

#include "cuda_img.h"

__global__ void kernel_select_insert(CudaImg img_big, CudaImg img_small, int2 pos, bool select)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_small.m_size.x || x + pos.x >= img_big.m_size.x) return;
    if (y >= img_small.m_size.y || y + pos.x >= img_big.m_size.y) return;

    if (select)
    {
        img_small.at4(y, x) = img_big.at4(pos.y + y, pos.x + x);
    }
    else
    {
    	uchar4 big_pixel = img_big.at4(pos.y + y, pos.x + x);
    	uchar4 small_pixel = img_small.at4(y, x);
    	uchar4 new_pixel;

    	new_pixel.x = small_pixel.x * small_pixel.w / 255 + big_pixel.x * ( 255 - small_pixel.w ) / 255;
    	new_pixel.y = small_pixel.y * small_pixel.w / 255 + big_pixel.y * ( 255 - small_pixel.w ) / 255;
    	new_pixel.z = small_pixel.z * small_pixel.w / 255 + big_pixel.z * ( 255 - small_pixel.w ) / 255;
    	new_pixel.w = big_pixel.w < small_pixel.w ? big_pixel.w : small_pixel.w;

    	img_big.at4(pos.y + y, pos.x + x) = new_pixel;
    }
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

    img_rotated.at4(newY, newX) = img_orig.at4(y, x);
}

__global__ void kernel_decrease_res(CudaImg img_orig, CudaImg img_small )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= img_small.m_size.x) return;
	if (y >= img_small.m_size.y) return;

	uint4 newColorTemp = {0,0,0,0};
	uchar4 pixel;
	for(int x2 = 0; x2 < 2; x2++)
	{
		for(int y2 = 0; y2 < 2; y2++)
		{
			pixel = img_orig.at4(2*y+y2, 2*x+x2);

			newColorTemp.x += pixel.x;
			newColorTemp.y += pixel.y;
			newColorTemp.z += pixel.z;
			newColorTemp.w += pixel.w;
		}
	}

	pixel.x = newColorTemp.x / 4;
	pixel.y = newColorTemp.y / 4;
	pixel.z = newColorTemp.z / 4;
	pixel.w = newColorTemp.w / 4;

	img_small.at4(y, x) = pixel;
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

void cu_decrease_res( CudaImg &img_orig, CudaImg &img_small )
{
	cudaError_t cudaErr;

	int block_size = 16;
	dim3 block_count;
	block_count.x = img_small.m_size.x / block_size + (img_small.m_size.x % block_size ? 1 : 0);
	block_count.y = img_small.m_size.y / block_size + (img_small.m_size.y % block_size ? 1 : 0);

	kernel_decrease_res<<<block_count, dim3(block_size, block_size)>>>(img_orig, img_small);

	if ((cudaErr = cudaGetLastError()) != cudaSuccess)
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );

	cudaDeviceSynchronize();
}
