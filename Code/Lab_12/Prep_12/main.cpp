#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

// 0 - Select, 1 - Insert, 2 - Rotate Clockwise, 3 - Rotate Counterclockwise
#define TEST 0

void cu_select_insert( CudaImg &t_big_rgb, CudaImg &t_small_rgb, int2 t_pos, bool select );
void cu_rotate_90( CudaImg &t_in_rgb, CudaImg &t_rot_rgb, bool clockwise );

int main(int argc, char** argv)
{
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (TEST == 0 || TEST == 1)
    {
        int2 size = {205, 407};
        int2 pos = {13, 7};

        cv::Mat imageSmall, imageBig;

        if (TEST == 0)
            imageSmall = cv::Mat(size.y, size.x, CV_8UC3);
        else
            imageSmall = cv::imread("imgSmall.png", cv::IMREAD_COLOR);
    
        imageBig = cv::imread("imgBig.png", cv::IMREAD_COLOR);
        

        CudaImg cudaImgSmall, cudaImgBig;
        cudaImgSmall.m_size.x = imageSmall.cols;
        cudaImgSmall.m_size.y = imageSmall.rows;
        cudaImgSmall.m_p_uchar3 = (uchar3*)imageSmall.data;
        cudaImgBig.m_size.x = imageBig.cols;
        cudaImgBig.m_size.y = imageBig.rows;
        cudaImgBig.m_p_uchar3 = (uchar3*)imageBig.data;

        cu_select_insert(cudaImgBig, cudaImgSmall, pos, TEST == 0);

        if (TEST == 0)
            cv::imshow("Selected image", imageSmall);
        else
            cv::imshow("Image after insertion", imageBig);
    }
    else if (TEST == 2 || TEST == 3)
    {
        cv::Mat image = cv::imread("imgBig.png", cv::IMREAD_COLOR);
        cv::Mat rotatedImage(image.cols, image.rows, CV_8UC3);

        CudaImg cudaImg, cudaImgRotated;
        cudaImg.m_size.x = image.cols;
        cudaImg.m_size.y = image.rows;
        cudaImg.m_p_uchar3 = (uchar3*)image.data;
        cudaImgRotated.m_size.x = rotatedImage.cols;
        cudaImgRotated.m_size.y = rotatedImage.rows;
        cudaImgRotated.m_p_uchar3 = (uchar3*)rotatedImage.data;

        cu_rotate_90(cudaImg, cudaImgRotated, TEST == 2);
        
        cv::imshow("Rotated image", rotatedImage);
    }
    else
        return 1;



    cv::waitKey(0);

    return 0;
}