#pragma once

#include "cuda_img.h"
#include "cudaErrCheck.h"

void cu_select_insert_internal( CudaImg &img_big, CudaImg &img_small, int2 pos, bool select );
void cu_rotate_90_internal( CudaImg &img_orig, CudaImg &img_rotated, bool clockwise );
void cu_decrease_res(CudaImg &img_orig, CudaImg &img_small);
void cu_decrease_res_internal( CudaImg &img_orig, CudaImg &img_small );

// Add flip wrapper
// Add blur wrapper