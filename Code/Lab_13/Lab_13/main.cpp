#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "cuda_img.h"
#include "Animation.h"

using namespace std;

int main(void)
{
	cv::Mat img_background= cv::imread("background.png", cv::IMREAD_UNCHANGED);
	CudaImg img_background_cuda = CudaImg(img_background);

	cv::Mat img_insert = cv::imread("insert.png", cv::IMREAD_UNCHANGED);
	CudaImg img_insert_cuda = CudaImg(img_insert);

    cv::Mat img_out(img_background_cuda.m_size.y, img_background_cuda.m_size.x, CV_8UC4);
    CudaImg img_out_cuda = CudaImg(img_out);


    Animation animation(img_background_cuda, img_insert_cuda);

    /*
    cv::Mat image_temp(img_background_cuda.m_size.y / 2, img_background_cuda.m_size.x, CV_8UC4);
    CudaImg image_temp_cuda(image_temp);

    animation.GibImg(image_temp_cuda, true);
    cv::imshow("Upper", image_temp);

    animation.GibImg(image_temp_cuda, false);
    cv::imshow("Lower", image_temp);
    */

    while(true)
    {
        if (cv::waitKey(1) != -1)
            break;
        
        animation.Step(img_out_cuda);

        cv::imshow("Animation", img_out);
    }

    return 0;
}
