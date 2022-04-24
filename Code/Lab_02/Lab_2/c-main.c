//***************************************************************************
//
// Program for education in subject "Assembly Languages"
// petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
//
// Empty project
//
//***************************************************************************

#include <stdio.h>

long g_long = 0x0807060504030201;
unsigned char g_longsum;
void sum_long();

long g_long2 = 0x4040404040404040;
int g_long2sum;
void sum_long2();

int g_int_arr[] = {100, -100, 205, -250, 1001, 8026, -7500, 903, 805, -1007};
int g_int_index_min;
void find_min();

int g_evensum;
void sum_even();

int main()
{
	sum_long();
	printf("g_longsum = %d\n", g_longsum);

	sum_long2();
	printf("g_long2sum = %d\n", g_long2sum);

	find_min();
	printf("g_int_index_min = %d\n", g_int_index_min);

	sum_even();
	printf("g_evensum = %d\n", g_evensum);

	return 0;
}
