#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

using namespace std;

void cu_select_insert( CudaImg &img_big, CudaImg &img_small, int2 pos, bool select );
void cu_rotate_90( CudaImg &img_orig, CudaImg &img_rotated, bool clockwise );
void cu_decrease_res( CudaImg &img_orig, CudaImg &img_small );

int main(void)
{
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator(&allocator);

	cv::Mat img_orig = cv::imread("image.png", cv::IMREAD_UNCHANGED);
	CudaImg img_orig_cuda = CudaImg(img_orig);

	cv::imshow("Original image", img_orig);



	cv::Mat img_final = cv::imread("image2.png", cv::IMREAD_UNCHANGED);
	CudaImg img_final_cuda = CudaImg(img_final);

	int2 pos = {4, 52};
	int2 size = {367, 277};


	cv::Mat img_cropped = cv::Mat(size.y, size.x, CV_8UC4);

	CudaImg img_cropped_cuda = CudaImg(img_cropped);

	cu_select_insert(img_orig_cuda, img_cropped_cuda, pos, true);

	cv::imshow("Cropped image", img_cropped);


	cv::Mat img_dec = cv::Mat(img_cropped.rows / 2, img_cropped.cols / 2, CV_8UC4);
	CudaImg img_dec_cuda = CudaImg(img_dec);

	cu_decrease_res(img_cropped_cuda, img_dec_cuda);

	cv::imshow("Decreased resolution", img_dec);


	cv::Mat img_dec_rotl = cv::Mat(img_dec.cols, img_dec.rows, CV_8UC4);
	cv::Mat img_dec_rotr = cv::Mat(img_dec.cols, img_dec.rows, CV_8UC4);

	CudaImg img_dec_rotl_cuda = CudaImg(img_dec_rotl);
	CudaImg img_dec_rotr_cuda = CudaImg(img_dec_rotr);

	cu_rotate_90(img_dec_cuda, img_dec_rotl_cuda, false);
	cu_rotate_90(img_dec_cuda, img_dec_rotr_cuda, true);


	int2 pos1 = {581, 86};
	int2 pos2 = {719, 254};

	cu_select_insert(img_final_cuda, img_dec_rotl_cuda, pos1, false);
	cu_select_insert(img_final_cuda, img_dec_rotr_cuda, pos2, false);

	cv::imshow("Final image", img_final);


    cv::waitKey(0);

    return 0;
}

/*
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
*/
