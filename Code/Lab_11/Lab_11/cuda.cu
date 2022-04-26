#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_split( CudaImg orig_img, CudaImg r_img, CudaImg g_img, CudaImg b_img)
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= orig_img.m_size.y ) return;
    if ( l_x >= orig_img.m_size.x ) return;

    uchar3 l_bgr = orig_img.at3(l_y, l_x);
    uchar3& l_r = r_img.at3(l_y, l_x);
    uchar3& l_g = g_img.at3(l_y, l_x);
    uchar3& l_b = b_img.at3(l_y, l_x);

    r_img.at3(l_y, l_x).z = l_bgr.z;
    g_img.at3(l_y, l_x).y = l_bgr.y;
    b_img.at3(l_y, l_x).x = l_bgr.x;
}

__global__ void kernel_dim( CudaImg t_img, uchar3 brightness)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_img.m_size.y ) return;
    if ( l_x >= t_img.m_size.x ) return;

    // Get point from color picture
    uchar3& l_bgr = t_img.at3(l_y, l_x);

    l_bgr.x = l_bgr.x * brightness.z / 100;
    l_bgr.y = l_bgr.y * brightness.y / 100;
    l_bgr.z = l_bgr.z * brightness.x / 100;
}

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;

    // Get point from color picture
    uchar3 l_bgr = t_color_cuda_img.at3(l_y, l_x);

    // Store BW point to new image
    t_bw_cuda_img.at1(l_y, l_x).x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
}

void cu_run_split( CudaImg orig_img, CudaImg r_img, CudaImg g_img, CudaImg b_img )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( orig_img.m_size.x + l_block_size - 1 ) / l_block_size, ( orig_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_split<<< l_blocks, l_threads >>>( orig_img, r_img, g_img, b_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}

void cu_run_dim( CudaImg t_img, uchar3 brightness )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_dim<<< l_blocks, l_threads >>>( t_img, brightness );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}

void cu_run_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_grayscale<<< l_blocks, l_threads >>>( t_color_cuda_img, t_bw_cuda_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}
