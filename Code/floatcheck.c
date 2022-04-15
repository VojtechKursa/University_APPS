#include <stdio.h>

#define ROUNDVERSION 0

int myRound(float x)
{
    int res = (int)x;
    if(x - res > 0.5)
        res++;

    return res;
}

int main(void)
{
    printf("Failed conversions:\n");

    int failed = 0;
    float f1, f2;
    int final;

    for (int i = 0; i <= 25000; i++)
    {
        f1 = i / (float)100;
        f2 = f1 * 100;
        final = ROUNDVERSION ? myRound(f2) : (int)f2;

        if(final != i)
        {
            printf("\t%d / 100 = %f * 100 = %f -> %d\n", i, f1, f2, final);
            failed++;
        }
    }

    printf("\nTotal: %d\n", failed);


    int i = 16379;
    float f = 163.79;
    printf("\n\n%d -> %f * 100 = %f -> %d\n", i, f, f*100, (int)(f*100));

    return 0;
}