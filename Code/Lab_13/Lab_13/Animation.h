#pragma once

#include <sys/time.h>
#include <iostream>

#include "cuda_img.h"
#include "cuda.h"
#include "insertProperties.h"

class Animation
{
private:
    CudaImg *img_background_lower = nullptr;
    CudaImg *img_background_upper = nullptr;
    CudaImg *img_inserted = nullptr;
    CudaImg *img_result = nullptr;

    int backgroundSpeed = 50;
    double upper_posY;
    double lower_posY;

    InsertProperties inserts[3];

    timeval lastFrameTime;
    timeval frameTreshold;

    timeval DoTiming();
    void DoMovementCalculation(timeval timeSinceLastFrame);

public:
    Animation(CudaImg img_background, CudaImg img_inserted);
    ~Animation();

    void Step(CudaImg img_result);

    void GibImg(CudaImg img, bool upper);
};
