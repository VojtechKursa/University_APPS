#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>

#define N 2000

void cu_add_arrays(int* arrIn1, int* arrIn2, int* arrOut, int length);

int main(void)
{
    int arr1[N], arr2[N], arrOut[N];

    for(int i = 0; i < N; i++)
    {
        arr1[i] = arr2[i] = i;
    }

    cu_add_arrays(arr1, arr2, arrOut, N);

    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", arr1[i], arr2[i], arrOut[i]);
    }

    return 0;
}