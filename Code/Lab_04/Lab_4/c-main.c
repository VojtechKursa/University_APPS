//***************************************************************************
//
// Program for education in subject "Assembly Languages"
// petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
//
// Empty project
//
//***************************************************************************

#include <stdio.h>

int even_nibbles(int* arr, int n);
int under_avg(long* arr, int n);
int most_common(int* arr, int n);
int shorter_vector(int* v1, int* v2);

int main()
{
	//	1.
	int arr_int[] = {0x59, 0x70, 0x1, 0x80, 0xFF, 50, 50, 100, 150, 300, 350, 350, 350, 400, 450, 450, 450, 500, 1000, 600, 450};
	printf("%d\n\n", even_nibbles(arr_int, 21));	//0x80 = 128

	//	2.
	long arr_long[] = {500, 200, 1, 9, 600, 593, 250};
	printf("%d\n\n", under_avg(arr_long, 7));	//4

	//	3.
	printf("%d\n\n", most_common(arr_int, 21));		//450

	//	4.
	int v1[] = {2,1,3};
	int v2[] = {2,1,2};
	printf("%d\n\n", shorter_vector(v1, v2));	//v2

	return 0;
}
