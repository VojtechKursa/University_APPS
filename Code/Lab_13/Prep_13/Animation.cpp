#include "Animation.h"

Animation::Animation(CudaImg img_background, CudaImg img_inserted, int2 startingPosition, int2 startingSpeed)
{
    this->lastFrameTime.tv_usec = 0;
    this->frameTreshold.tv_usec = 15000;

    this->insert_position = startingPosition;
    this->insert_speed = startingSpeed;

    this->img_background = new CudaImg(img_background.m_size);
    cudaErrCheck(cudaMalloc(&(this->img_background->m_p_void), img_background.m_size.x * img_background.m_size.y * 4));
    cudaErrCheck(cudaMemcpy(this->img_background->m_p_void, img_background.m_p_void, img_background.m_size.x * img_background.m_size.y * 4, cudaMemcpyHostToDevice));

    this->img_inserted = new CudaImg(img_inserted.m_size);
    cudaErrCheck(cudaMalloc(&(this->img_inserted->m_p_void), img_inserted.m_size.x * img_inserted.m_size.y * 4)); 
    cudaErrCheck(cudaMemcpy(this->img_inserted->m_p_void, img_inserted.m_p_void, img_inserted.m_size.x * img_inserted.m_size.y * 4, cudaMemcpyHostToDevice));

    this->img_inserted_flipped = new CudaImg(img_inserted.m_size);
    cudaErrCheck(cudaMalloc(&(this->img_inserted_flipped->m_p_void), img_inserted_flipped->m_size.x * img_inserted_flipped->m_size.y * 4));
    cu_flip_internal(*this->img_inserted, *this->img_inserted_flipped);

    this->img_insert_blurred = new CudaImg(img_inserted.m_size);
    cudaErrCheck(cudaMalloc(&(this->img_insert_blurred->m_p_void), img_insert_blurred->m_size.x * img_insert_blurred->m_size.y * 4));

    this->img_result = new CudaImg(img_background.m_size);
    cudaErrCheck(cudaMalloc(&(this->img_result->m_p_void), img_result->m_size.x * img_result->m_size.y * 4));
}

Animation::~Animation()
{
    if (img_background != nullptr)
    {
        cudaErrCheck(cudaFree(img_background->m_p_void));
        delete img_background;
    }
    if (img_inserted != nullptr)
    {
        cudaErrCheck(cudaFree(img_inserted->m_p_void));
        delete img_inserted;
    }
    if (img_inserted_flipped != nullptr)
    {
        cudaErrCheck(cudaFree(img_inserted_flipped->m_p_void));
        delete img_inserted_flipped;
    }
    if (img_insert_blurred != nullptr)
    {
        cudaErrCheck(cudaFree(img_insert_blurred->m_p_void));
        delete img_insert_blurred;
    }
    if (img_result != nullptr)
    {
        cudaErrCheck(cudaFree(img_result->m_p_void));
        delete img_result;
    }
}



void Animation::Step(CudaImg img_result)
{
    timeval timeSinceLast = DoTiming();

    if (timeSinceLast.tv_usec >= this->frameTreshold.tv_usec)
    {
        DoMovementCalculation(timeSinceLast);


        cudaErrCheck(cudaMemcpy(this->img_result->m_p_void, this->img_background->m_p_void, this->img_background->m_size.x * this->img_background->m_size.y * 4, cudaMemcpyDeviceToDevice));

        CudaImg& imageToInsert = *(this->insert_speed.x < 0 ? this->img_inserted_flipped : this->img_inserted);
        cu_blur_internal(imageToInsert, *this->img_insert_blurred, blur);

        cu_select_insert_internal(*this->img_result, *this->img_insert_blurred, this->insert_position, false);
    }

    cudaErrCheck(cudaMemcpy(img_result.m_p_void, this->img_result->m_p_void, img_result.m_size.x * img_result.m_size.y * 4, cudaMemcpyDeviceToHost));
}



timeval Animation::DoTiming()
{
    timeval currTime, timeSinceLastFrame;

    gettimeofday(&currTime, NULL);

    if (this->lastFrameTime.tv_usec == 0)
    {
        timeSinceLastFrame.tv_usec = 0;
        this->lastFrameTime = currTime;
    }
    else
        timersub(&currTime, &this->lastFrameTime, &timeSinceLastFrame);

    if (timeSinceLastFrame.tv_usec >= frameTreshold.tv_usec)
    {
        this->lastFrameTime = currTime;
    }

    return timeSinceLastFrame;
}

void Animation::DoMovementCalculation(timeval timeSinceLastFrame)
{
    double seconds = timeSinceLastFrame.tv_usec / (double)1000000;
    
    this->insert_position.x += round(this->insert_speed.x * seconds);
    this->insert_position.y += round(this->insert_speed.y * seconds);

    this->blurBuffer += this->blurChange * seconds;
    if (this->blurBuffer >= 1 || this->blurBuffer <= -1)
    {
        this->blur += (int)this->blurBuffer;
        this->blurBuffer -= (int)this->blurBuffer;
    }

    bool adjustMade;
    do
    {
        adjustMade = MovementOverflowGuard();
    } while (adjustMade);
}

bool Animation::MovementOverflowGuard()
{
    bool adjustMade = false;

    if(this->insert_position.x < 0)
    {
        this->insert_position.x *= -1;
        this->insert_speed.x *= -1;

        adjustMade = true;
    }
    else if(this->insert_position.x + this->img_inserted->m_size.x > img_background->m_size.x)
    {
        this->insert_position.x -= 2*((this->insert_position.x + this->img_inserted->m_size.x) - this->img_background->m_size.x);
        this->insert_speed.x *= -1;

        adjustMade = true;
    }
    
    if(this->insert_position.y < 0)
    {
        this->insert_position.y *= -1;
        this->insert_speed.y *= -1;

        adjustMade = true;
    }
    else if(this->insert_position.y + this->img_inserted->m_size.y > img_background->m_size.y)
    {
        this->insert_position.y -= 2*((this->insert_position.y + this->img_inserted->m_size.y) - this->img_background->m_size.y);
        this->insert_speed.y *= -1;
        
        adjustMade = true;
    }

    if(this->blur < 1)
    {
        if(this->blur == 0)
            this->blur = -1;
        
        this->blur *= -1;
        this->blurChange *= -1;

        adjustMade = true;
    }
    else if(this->blur > this->blurLimit)
    {
        this->blur = this->blurLimit - (this->blur - this->blurLimit);
        this->blurChange *= -1;

        adjustMade = true;
    }

    return adjustMade;
}