//***************************************************************************
//
// Program for education in subject "Assembly Languages"
// petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
//
// Empty project
//
//***************************************************************************

#include <stdio.h>

int g_int_a, g_int_b;
void set_int();

void switch_ints();

char g_char_arr[5];
void load_g_char_arr();

char g_char_str[] = "Puma je kocka.";
void edit_g_char_str();

char g_char_num[2] = {111, -123};
void move_g_char_num();

int main()
{
	set_int();
	printf("g_int_a = %d\ng_int_b = 0x%x\n", g_int_a, g_int_b);

	switch_ints();
	printf("\ng_int_a = 0x%x\ng_int_b = %d\n", g_int_a, g_int_b);

	load_g_char_arr();
	printf("\ng_char_arr = %s\n", g_char_arr);

	edit_g_char_str();
	printf("\ng_char_str = %s\n", g_char_str);

	move_g_char_num();
	printf("\ng_int_a = %d\ng_int_b = %d\n", g_int_a, g_int_b);

	return 0;
}
