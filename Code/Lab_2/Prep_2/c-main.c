//***************************************************************************
//
// Program for education in subject "Assembly Languages"
// petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
//
// Empty project
//
//***************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int g_int_arr[ 10 ] = { 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000 };
int g_int_suma, g_int_licha, g_int_ones;
long g_long_num = 0xCAFEACACA0;
char g_char_str[] = "Kratky zbytecny t e x t.";
int g_int_length, g_int_spaces;

void sum_int_arr();
void negate_int_arr();
void count_odd();
void count_ones();
void str_length();
void count_spaces();

void print_arr()
{
	for(int i = 0; i < 10; i++)
	{
		printf("g_int_arr[%d] = %d\n", i, g_int_arr[i]);
	}
}

void randomize_arr()
{
	srand(time(NULL));

	for(int i = 0; i < 10; i++)
	{
		g_int_arr[i] = rand();
	}
}

int main()
{
	sum_int_arr();
	printf("sum_int_arr = %d\n\n", g_int_suma);

	negate_int_arr();
	print_arr();
	printf("\n");

	randomize_arr();
	print_arr();
	count_odd();
	printf("g_int_licha = %d\n\n", g_int_licha);

	count_ones();
	printf("g_int_ones = %d\n\n", g_int_ones);

	str_length();
	printf("g_int_length = %d\n\n", g_int_length);

	count_spaces();
	printf("g_int_spaces = %d\n\n", g_int_spaces);

	return 0;
}
