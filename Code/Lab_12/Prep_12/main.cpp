#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

using namespace std;

void cu_select_insert( CudaImg &img_big, CudaImg &img_small, int2 pos, bool select );
void cu_rotate_90( CudaImg &img_orig, CudaImg &img_rotated, bool clockwise );

int main(int argc, char** argv)
{
    // 0 - Select, 1 - Insert, 2 - Rotate Clockwise, 3 - Rotate Counterclockwise
    int test_num;

    if(argc < 2)
    {
        cout << "ERR: Test number not specified.\n";
        exit(1);
    }

    test_num = argv[1][0] - '0';

    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (test_num == 0 || test_num == 1)
    {
        int2 size = {205, 407};
        int2 pos = {13, 7};

        cv::Mat img_small, img_big;

        if (test_num == 0)
            img_small = cv::Mat(size.y, size.x, CV_8UC3);
        else
            img_small = cv::imread("imgSmall.png", cv::IMREAD_COLOR);
    
        img_big = cv::imread("imgBig.png", cv::IMREAD_COLOR);
        

        CudaImg img_small_cuda(img_small);
        CudaImg img_big_cuda(img_big);

        cu_select_insert(img_big_cuda, img_small_cuda, pos, test_num == 0);

        if (test_num == 0)
            cv::imshow("Selected image", img_small);
        else
            cv::imshow("Image after insertion", img_big);
    }
    else if (test_num == 2 || test_num == 3)
    {
        cv::Mat img_orig = cv::imread("imgBig.png", cv::IMREAD_COLOR);
        cv::Mat img_rotated(img_orig.cols, img_orig.rows, CV_8UC3);

        CudaImg img_orig_cuda(img_orig);
        CudaImg img_rotated_cuda(img_rotated);

        cu_rotate_90(img_orig_cuda, img_rotated_cuda, test_num == 2);
        
        cv::imshow("Rotated image", img_rotated);
    }
    else
    {
        cout << "ERR: Invalid test number specified.\n";
        exit(1);
    }



    cv::waitKey(0);

    return 0;
}