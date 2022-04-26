#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

// Function prototype from .cu file
void cu_run_split(CudaImg img_orig, CudaImg img_r, CudaImg img_g, CudaImg img_b);
void cu_run_dim( CudaImg img, uchar3 brightness );
void cu_run_grayscale( CudaImg img_color, CudaImg img_greyscale );


int main( int argc, char **argv )
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator( &allocator );

    if ( argc < 2 )
    {
        printf( "Enter picture filename!\n" );
        return 1;
    }

    // Originally loaded image
    cv::Mat img_orig = cv::imread( argv[ 1 ], cv::IMREAD_COLOR );

    if ( !img_orig.data )
    {
        printf( "Unable to read file '%s'\n", argv[ 1 ] );
        return 1;
    }

    cv::Mat img_bw( img_orig.size(), CV_8U );

    CudaImg img_orig_cuda(img_orig);
    CudaImg img_bw_cuda(img_bw);

    cu_run_grayscale( img_orig_cuda, img_bw_cuda );

    cv::imshow( "Color", img_orig );
    cv::imshow( "GrayScale", img_bw );


    // Split

	cv::Mat img_r(img_orig.size(), CV_8UC3);
	cv::Mat img_g(img_orig.size(), CV_8UC3);
	cv::Mat img_b(img_orig.size(), CV_8UC3);

	CudaImg img_r_cuda(img_r);
	CudaImg img_g_cuda(img_g);
	CudaImg img_b_cuda(img_b);

	cu_run_split(img_orig_cuda, img_r_cuda, img_g_cuda, img_b_cuda);

	cv::imshow("Red channel", img_r);
	cv::imshow("Green channel", img_g);
	cv::imshow("Blue channel", img_b);


    // Dim

    cv::Mat img_br = cv::Mat(img_orig);
    CudaImg img_br_cuda(img_br);

    uchar3 brightness;
    brightness.x = 100;
    brightness.y = 50;
    brightness.z = 0;

    cu_run_dim(img_br_cuda, brightness);

    cv::imshow("Dimmed", img_br);


    cv::waitKey( 0 );

    return 0;
}
