// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <opencv2/core/mat.hpp>

class CudaImg
{
public:
	uint3 m_size; // size of picture
	union
	{
		void *m_p_void;		// data of picture
		uchar1 *m_p_uchar1; // data of picture
		uchar3 *m_p_uchar3; // data of picture
		uchar4 *m_p_uchar4; // data of picture
	};

	CudaImg(cv::Mat matrix);
	CudaImg(uint3 size);

	/*
	GPU code included in header file, because nvcc doesn't like
	when functions used in one file are just defined by header and
	implemented in another file and I can't be bothered to research a way to
	tell it to link everything together properly
	*/

	__device__ uchar1& at1(int y, int x)
	{
		return m_p_uchar1[y * m_size.x + x];
	}

	__device__ uchar3& at3(int y, int x)
	{
		return m_p_uchar3[y * m_size.x + x];
	}

	__device__ uchar4& at4(int y, int x)
	{
	  return m_p_uchar4[y * m_size.x + x];
	}
};
