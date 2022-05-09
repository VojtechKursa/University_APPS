#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "cuda_img.h"
#include "Animation.h"

using namespace std;

int main(void)
{
	cv::Mat img_background_full = cv::imread("background.png", cv::IMREAD_UNCHANGED);
	CudaImg img_background_full_cuda = CudaImg(img_background_full);

	cv::Mat img_insert_full = cv::imread("insert.png", cv::IMREAD_UNCHANGED);
	CudaImg img_insert_full_cuda = CudaImg(img_insert_full);
    
    cv::Mat img_background(img_background_full_cuda.m_size.y / 2, img_background_full_cuda.m_size.x / 2, CV_8UC4);
	CudaImg img_background_cuda = CudaImg(img_background);

	cv::Mat img_insert(img_insert_full_cuda.m_size.y / 2, img_insert_full_cuda.m_size.x / 2, CV_8UC4);
	CudaImg img_insert_cuda = CudaImg(img_insert);

    cv::Mat img_out(img_background_cuda.m_size.y, img_background_cuda.m_size.x, CV_8UC4);
    CudaImg img_out_cuda = CudaImg(img_out);


    cu_decrease_res(img_background_full_cuda, img_background_cuda);
    cu_decrease_res(img_insert_full_cuda, img_insert_cuda);


    int2 initialPos, initialSpeed;
    initialPos.x = 10;
    initialPos.y = (img_background_cuda.m_size.y / 2) - (img_insert_cuda.m_size.y / 2);
    initialSpeed.x = (img_background_cuda.m_size.x - img_insert_cuda.m_size.x) / 5;
    initialSpeed.y = -((img_background_cuda.m_size.y - img_insert_cuda.m_size.y) / 10);

    Animation animation(img_background_cuda, img_insert_cuda, initialPos, initialSpeed);

    while(true)
    {
        if (cv::waitKey(1) != -1)
            break;
        
        animation.Step(img_out_cuda);

        cv::imshow("Animation", img_out);
    }

    return 0;
}
