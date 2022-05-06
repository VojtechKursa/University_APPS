#pragma once

#include <sys/time.h>
#include <iostream>

#include "cuda_img.h"
#include "cuda.h"

class Animation
{
private:
    CudaImg *img_background = nullptr;
    CudaImg *img_inserted = nullptr;
    CudaImg *img_inserted_flipped = nullptr;
    CudaImg *img_insert_blurred = nullptr;
    CudaImg *img_result = nullptr;

    timeval lastFrameTime;
    timeval frameTreshold;
    
    int2 insert_position;
    int2 insert_speed;  //!< In pixels/s

    uint8_t blur = 1;
    uint8_t blurLimit = 5;
    double blurBuffer = 0;
    double blurChange = 1;

    timeval DoTiming();
    void DoMovementCalculation(timeval timeSinceLastFrame);
    bool MovementOverflowGuard();

public:
    //!\param startingSpeed In pixels/s
    Animation(CudaImg img_background, CudaImg img_inserted, int2 startingPosition, int2 startingSpeed);
    ~Animation();

    void Step(CudaImg img_result);
};