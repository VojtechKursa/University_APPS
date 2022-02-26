//***************************************************************************
//
// Program for education in subject "Assembly Languages"
// petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
//
// Empty project
//
//***************************************************************************

#include <stdio.h>

int max_deviation( int *t_array, int t_len );
long max_bits_one( long *t_array, int t_len );
int to_num( char *t_str );
int in_circle( int t_x, int t_y, int t_R );
char digit( char *t_str );
char digit_counter[] = {0,0,0,0,0,0,0,0,0,0};

int main()
{
	// 1.
	int int_arr[] = {1,2,3,4,5,6,7,8,9,15};
	printf("%d\n\n", max_deviation(int_arr, 10));

	// 2.
	long long_arr[] = {0xFF, 0x050, 0x87, 0xFFFF, 0x8787};
	printf("%ld\n\n", max_bits_one(long_arr, 5));

	// 3.
	char num_str[] = "-121381559";
	printf("%d\n\n", to_num(num_str));

	// 4.
	printf("%d\n\n", in_circle(1,1,2));

	// 5.
	printf("%c\n\n", digit(num_str));

	return 0;
}
