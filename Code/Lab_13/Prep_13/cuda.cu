#include "cuda.h"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <iostream>

#include "cuda_kernels.h"

void cu_select_insert_internal( CudaImg &img_big, CudaImg &img_small, int2 pos, bool select )
{
    cudaError_t cudaErr;

    int block_size = 16;
    dim3 block_count;
    block_count.x = img_small.m_size.x / block_size + (img_small.m_size.x % block_size ? 1 : 0);
    block_count.y = img_small.m_size.y / block_size + (img_small.m_size.y % block_size ? 1 : 0);

    kernel_select_insert<<<block_count, dim3(block_size, block_size)>>>(img_big, img_small, pos, select);

    if ((cudaErr = cudaGetLastError()) != cudaSuccess)
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );
}

void cu_rotate_90_internal( CudaImg &img_orig, CudaImg &img_rotated, bool clockwise )
{
    cudaError_t cudaErr;

    int block_size = 16;
    dim3 block_count;
    block_count.x = img_orig.m_size.x / block_size + (img_orig.m_size.x % block_size ? 1 : 0);
    block_count.y = img_orig.m_size.y / block_size + (img_orig.m_size.y % block_size ? 1 : 0);

    kernel_rotate_90<<<block_count, dim3(block_size, block_size)>>>(img_orig, img_rotated, clockwise);

    if ((cudaErr = cudaGetLastError()) != cudaSuccess)
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );
}

void cu_decrease_res(CudaImg &img_orig, CudaImg &img_small)
{
    CudaImg img_orig_internal = img_orig.m_size;
    CudaImg img_small_internal = img_small.m_size;

    cudaErrCheck(cudaMalloc(&img_orig_internal.m_p_void, img_orig.m_size.x * img_orig.m_size.y * 4));
    cudaErrCheck(cudaMemcpy(img_orig_internal.m_p_void, img_orig.m_p_void, img_orig.m_size.x * img_orig.m_size.y * 4, cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc(&img_small_internal.m_p_void, img_small_internal.m_size.x * img_small_internal.m_size.y * 4));
    

    cu_decrease_res_internal(img_orig_internal, img_small_internal);


    cudaErrCheck(cudaMemcpy(img_small.m_p_void, img_small_internal.m_p_void, img_small.m_size.x * img_small.m_size.y * 4, cudaMemcpyDeviceToHost));

    cudaErrCheck(cudaFree(img_orig_internal.m_p_void));
    cudaErrCheck(cudaFree(img_small_internal.m_p_void));
}

void cu_decrease_res_internal( CudaImg &img_orig, CudaImg &img_small )
{
	cudaError_t cudaErr;

	int block_size = 16;
	dim3 block_count;
	block_count.x = img_small.m_size.x / block_size + (img_small.m_size.x % block_size ? 1 : 0);
	block_count.y = img_small.m_size.y / block_size + (img_small.m_size.y % block_size ? 1 : 0);

	kernel_decrease_res<<<block_count, dim3(block_size, block_size)>>>(img_orig, img_small);

	if ((cudaErr = cudaGetLastError()) != cudaSuccess)
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );
}

void cu_flip_internal( CudaImg &img_orig, CudaImg &img_flipped)
{
    cudaError_t cudaErr;

    int block_size = 16;
    dim3 block_count;
    block_count.x = img_orig.m_size.x / block_size + (img_orig.m_size.x % block_size ? 1 : 0);
    block_count.y = img_orig.m_size.y / block_size + (img_orig.m_size.y % block_size ? 1 : 0);

    kernel_flip<<<block_count, dim3(block_size, block_size)>>>(img_orig, img_flipped);

    if ((cudaErr = cudaGetLastError()) != cudaSuccess)
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cudaErr ) );
}
