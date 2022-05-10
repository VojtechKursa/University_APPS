#pragma once

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_select_insert(CudaImg img_big, CudaImg img_small, int2 pos, bool select);
__global__ void kernel_rotate_90(CudaImg img_orig, CudaImg img_rotated, bool clockwise);
__global__ void kernel_decrease_res(CudaImg img_orig, CudaImg img_small );
__global__ void kernel_clear(CudaImg img);
