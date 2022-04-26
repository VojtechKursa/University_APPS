#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

// Function prototype from .cu file
void cu_run_grayscale( CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img );
void cu_run_dim( CudaImg t_img, uchar3 brightness );
void cu_run_split( CudaImg orig_img, CudaImg r_img, CudaImg g_img, CudaImg b_img );

int main( int t_numarg, char **t_arg )
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator( &allocator );

    if ( t_numarg < 2 )
    {
        printf( "Enter picture filename!\n" );
        return 1;
    }

    // Load image
    cv::Mat l_bgr_cv_img = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR ); // CV_LOAD_IMAGE_COLOR );

    if ( !l_bgr_cv_img.data )
    {
        printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
        return 1;
    }

    // create empty BW image
    cv::Mat l_bw_cv_img( l_bgr_cv_img.size(), CV_8U );

    // data for CUDA
    CudaImg l_bgr_cuda_img(l_bgr_cv_img);
    CudaImg l_bw_cuda_img(l_bw_cv_img);

    // Function calling from .cu file
    cu_run_grayscale( l_bgr_cuda_img, l_bw_cuda_img );

    // Show the Color and BW image
    cv::imshow( "Color", l_bgr_cv_img );
    cv::imshow( "GrayScale", l_bw_cv_img );


    // Split
	cv::Mat img_r(l_bgr_cv_img.size(), CV_8UC3);
	cv::Mat img_g(l_bgr_cv_img.size(), CV_8UC3);
	cv::Mat img_b(l_bgr_cv_img.size(), CV_8UC3);

	CudaImg img_r_cuda(img_r);
	CudaImg img_g_cuda(img_g);
	CudaImg img_b_cuda(img_b);

	cu_run_split(l_bgr_cuda_img, img_r_cuda, img_g_cuda, img_b_cuda);

	cv::imshow("Red channel", img_r);
	cv::imshow("Green channel", img_g);
	cv::imshow("Blue channel", img_b);


    // Dim
    cv::Mat img_br = cv::Mat(l_bgr_cv_img);
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
