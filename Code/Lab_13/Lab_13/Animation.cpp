#include "Animation.h"

Animation::Animation(CudaImg img_background, CudaImg img_inserted)
{
    this->lastFrameTime.tv_usec = 0;
    this->frameTreshold.tv_usec = 15000;


    uint3 halfSize = {img_background.m_size.x, img_background.m_size.y / 2, img_background.m_size.z};

    this->img_background_upper = new CudaImg(halfSize);
    this->img_background_lower = new CudaImg(halfSize);

    cudaErrCheck(cudaMalloc(&(this->img_background_upper->m_p_void), img_background_upper->m_size.x * img_background_upper->m_size.y * 4));
    cudaErrCheck(cudaMalloc(&(this->img_background_lower->m_p_void), img_background_lower->m_size.x * img_background_lower->m_size.y * 4));

    CudaImg img_background_int(img_background.m_size);
    cudaErrCheck(cudaMalloc(&(img_background_int.m_p_void), img_background_int.m_size.x * img_background_int.m_size.y * 4));
    cudaErrCheck(cudaMemcpy(img_background_int.m_p_void, img_background.m_p_void, img_background.m_size.x * img_background.m_size.y * 4, cudaMemcpyHostToDevice));

    cu_split_internal(img_background_int, *this->img_background_upper, *this->img_background_lower);

    cudaErrCheck(cudaFree(img_background_int.m_p_void));


    this->img_inserted = new CudaImg(img_inserted.m_size);
    cudaErrCheck(cudaMalloc(&(this->img_inserted->m_p_void), img_inserted.m_size.x * img_inserted.m_size.y * 4)); 
    cudaErrCheck(cudaMemcpy(this->img_inserted->m_p_void, img_inserted.m_p_void, img_inserted.m_size.x * img_inserted.m_size.y * 4, cudaMemcpyHostToDevice));

    this->img_result = new CudaImg(img_background.m_size);
    cudaErrCheck(cudaMalloc(&(this->img_result->m_p_void), img_result->m_size.x * img_result->m_size.y * 4));


    this->upper_posY = -1 * (int)(this->img_background_upper->m_size.y);
    this->lower_posY = this->img_result->m_size.y;

    this->inserts[0] = {250, -50, 20};
    this->inserts[1] = {0, -120, 40};
    this->inserts[2] = {100, -150, 60};
}

Animation::~Animation()
{
    if (img_background_upper != nullptr)
    {
        cudaErrCheck(cudaFree(img_background_upper->m_p_void));
        delete img_background_upper;
    }
    if (img_background_lower != nullptr)
    {
        cudaErrCheck(cudaFree(img_background_lower->m_p_void));
        delete img_background_lower;
    }
    if (img_inserted != nullptr)
    {
        cudaErrCheck(cudaFree(img_inserted->m_p_void));
        delete img_inserted;
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

        cu_clear_internal(*this->img_result);

        cu_select_insert_internal(*this->img_result, *this->img_background_upper, {0, (int)round(upper_posY)}, false);
        cu_select_insert_internal(*this->img_result, *this->img_background_lower, {0, (int)round(lower_posY)}, false);

        for(int i = 0; i < 3; i++)
        {
            cu_select_insert_internal(*this->img_result, *this->img_inserted, {(int)this->inserts[i].posX, (int)this->inserts[i].posY}, false);
        }
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

    upper_posY += seconds * backgroundSpeed;
    lower_posY -= seconds * backgroundSpeed;

    if(upper_posY + img_background_upper->m_size.y >= lower_posY)
    {
    	upper_posY = 0;
    	lower_posY = img_background_upper->m_size.y;
    }

    for(int i = 0; i < 3; i++)
    {
    	inserts[i].posY += seconds * inserts[i].speed;
    }
}

void Animation::GibImg(CudaImg img, bool upper)
{
	CudaImg selectedImage = *(upper ? img_background_upper : img_background_lower);

	cudaMemcpy(img.m_p_void, selectedImage.m_p_void, selectedImage.m_size.x * selectedImage.m_size.y * 4, cudaMemcpyDeviceToHost);
}
