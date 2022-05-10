#include "cuda_img.h"

CudaImg::CudaImg(cv::Mat matrix)
{
	m_size.x = matrix.cols;
	m_size.y = matrix.rows;
	m_size.z = matrix.channels();

	m_p_void = matrix.data;
}

CudaImg::CudaImg(uint3 size)
{
	m_size = size;

	m_p_void = nullptr;
}
