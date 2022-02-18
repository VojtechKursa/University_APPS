//***************************************************************************
//
// Program for education in subject "Assembly Languages"
// petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
//
// Empty project
//
//***************************************************************************

#include <stdio.h>

int move_int(int *t_in_array, int *t_out_array, int t_N);
int multiple_24 (int *t_array, int t_N);
long negative_max(long *t_array, int t_N);
unsigned char biggest_byte(long t_num);
int highest_bit(long t_num);
int replace_digit(char *t_str, char t_replace);

int main()
{
	// 1.
	int int_in_arr[10] = {1,-1,8,5,6,-2,-3,-8,9,10};
	int int_out_arr[10];

	int n = move_int(int_in_arr, int_out_arr, 10);

	for(int i = 0;i<n;i++)
	{
		printf("%d ", int_out_arr[i]);
	}
	printf("\n\n");

	// 2.
	printf("%d\n\n", multiple_24(int_in_arr, 10));

	// 3.
	long long_array[10] = {8, 0, 200, -800, 900, -1000, -600, 5, -200, 3};
	printf("%ld\n\n", negative_max(long_array, 10));

	// 4.
	long long_num = 0x09010F0C1008;
	printf("%d\n\n", (int)biggest_byte(long_num));

	// 5.
	printf("%d\n\n", highest_bit(long_num));

	// 6.
	char str[] = "H3ell0 w00r7d5";
	int replaced = replace_digit(str, 'X');
	printf("Replaced characters: %d\nResult string:%s\n\n", replaced, str);

	return 0;
}
