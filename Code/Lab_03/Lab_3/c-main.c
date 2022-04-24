//***************************************************************************
//
// Program for education in subject "Assembly Languages"
// petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
//
// Empty project
//
//***************************************************************************

#include <stdio.h>

long search_amax( long *t_array, int t_len );
int search_mask( int *t_array, int t_len, int t_bits );
void copy_nospace( const char *t_str_in, char *t_str_out );
long num_digits( const char *t_str );

int main()
{
	long long_arr[] = {5, -8, 500000, -800000, 7};
	printf("%ld\n\n", search_amax(long_arr, 5));

	int int_arr[] = {0x5, 0x4, 0x8, 0x80, 0xF8, 0xF1};
	printf("%d\n\n", search_mask(int_arr, 6, 3));

	const char str[] = " ab c  dx o59 .l 8x 5 p3a";
	char newstr[50];
	copy_nospace(str, newstr);
	printf("%s\n\n", newstr);

	printf("%ld\n\n", num_digits(str));

	return 0;
}
