#include "cudaErrCheck.h"

void cudaErrCheck(cudaError err)
{
    if (err != cudaSuccess)
    {
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
}