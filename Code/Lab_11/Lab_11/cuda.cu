#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_split(CudaImg img_orig, CudaImg img_r, CudaImg img_g, CudaImg img_b)
{
    int posX = blockDim.x * blockIdx.x + threadIdx.x;
	int posY = blockDim.y * blockIdx.y + threadIdx.y;
    if ( posX >= img_orig.m_size.x ) return;
    if ( posY >= img_orig.m_size.y ) return;

    uchar3 pixel_orig_bgr = img_orig.at3(posY, posX);
    uchar3& pixel_r_bgr = img_r.at3(posY, posX);
    uchar3& pixel_g_bgr = img_g.at3(posY, posX);
    uchar3& pixel_b_bgr = img_b.at3(posY, posX);

    pixel_r_bgr.z = pixel_orig_bgr.z;
    pixel_r_bgr.y = pixel_r_bgr.x = 0;

    pixel_g_bgr.y = pixel_orig_bgr.y;
    pixel_g_bgr.x = pixel_g_bgr.z = 0;
    
    pixel_b_bgr.x = pixel_orig_bgr.x;
    pixel_b_bgr.y = pixel_b_bgr.z = 0;
}

__global__ void kernel_dim(CudaImg img, uchar3 brightness)
{
    int posX = blockDim.x * blockIdx.x + threadIdx.x;
	int posY = blockDim.y * blockIdx.y + threadIdx.y;
    if ( posX >= img.m_size.x ) return;
    if ( posY >= img.m_size.y ) return;

    uchar3& pixel_bgr = img.at3(posY, posX);

    pixel_bgr.x = pixel_bgr.x * brightness.z / 100;
    pixel_bgr.y = pixel_bgr.y * brightness.y / 100;
    pixel_bgr.z = pixel_bgr.z * brightness.x / 100;
}

__global__ void kernel_grayscale(CudaImg img_color, CudaImg image_grayscale)
{
    int posX = blockDim.x * blockIdx.x + threadIdx.x;
	int posY = blockDim.y * blockIdx.y + threadIdx.y;
    if ( posX >= img_color.m_size.x ) return;
    if ( posY >= img_color.m_size.y ) return;

    uchar3 pixel_bgr = img_color.at3(posY, posX);

    // Store BW point to new image
    image_grayscale.at1(posY, posX).x = pixel_bgr.x * 0.11 + pixel_bgr.y * 0.59 + pixel_bgr.z * 0.30;
}



void cu_run_split(CudaImg img_orig, CudaImg img_r, CudaImg img_g, CudaImg img_b)
{
    cudaError_t cuda_err;

    int block_size = 16;

    dim3 block_count;
    block_count.x = img_orig.m_size.x / block_size + (img_orig.m_size.x % block_size ? 1 : 0);
    block_count.y = img_orig.m_size.y / block_size + (img_orig.m_size.y % block_size ? 1 : 0);
    
    dim3 thread_count( block_size, block_size );

    kernel_split<<< block_count, thread_count >>>( img_orig, img_r, img_g, img_b );

    if ( ( cuda_err = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cuda_err ) );

    cudaDeviceSynchronize();
}

void cu_run_dim( CudaImg img, uchar3 brightness )
{
    cudaError_t cuda_err;

    int block_size = 16;
    
    dim3 block_count;
    block_count.x = img.m_size.x / block_size + (img.m_size.x % block_size ? 1 : 0);
    block_count.y = img.m_size.y / block_size + (img.m_size.y % block_size ? 1 : 0);

    dim3 thread_count( block_size, block_size );

    kernel_dim<<< block_count, thread_count >>>( img, brightness );

    if ( ( cuda_err = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cuda_err ) );

    cudaDeviceSynchronize();
}

void cu_run_grayscale( CudaImg img_color, CudaImg img_greyscale )
{
    cudaError_t cuda_err;

    int block_size = 16;
    
    dim3 block_count;
    block_count.x = img_color.m_size.x / block_size + (img_color.m_size.x % block_size ? 1 : 0);
    block_count.y = img_color.m_size.y / block_size + (img_color.m_size.y % block_size ? 1 : 0);

    dim3 thread_count( block_size, block_size );

    kernel_grayscale<<< block_count, thread_count >>>( img_color, img_greyscale );

    if ( ( cuda_err = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cuda_err ) );

    cudaDeviceSynchronize();
}
