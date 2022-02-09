;***************************************************************************
;
; Program for education in subject "Assembly Languages" and "APPS"
; petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
;
; Empty project
;
;***************************************************************************

    bits 64

    section .data

	extern g_int_a, g_int_b
	extern g_char_arr, g_char_str
	extern g_char_num

;***************************************************************************

    section .text

	global set_int
set_int:
	mov [g_int_a], dword 123456
	mov [g_int_b], dword 0x33343536
	ret

	global switch_ints
switch_ints:
	mov r8d, [g_int_a]
	mov r9d, [g_int_b]
	mov [g_int_a], r9d
	mov [g_int_b], r8d
	ret

	global load_g_char_arr
load_g_char_arr:
	mov r8d, [g_int_a]
	mov [g_char_arr], r8d
	ret

	global edit_g_char_str
edit_g_char_str:
	mov [g_char_str], byte 'T'
	mov [g_char_str + 1], byte 'y'
	mov [g_char_str + 2], byte 'g'
	mov [g_char_str + 3], byte 'r'
	ret

	global move_g_char_num
move_g_char_num:
	movsx r8d, byte [g_char_num]
	mov [g_int_a], r8d
	movsx r8d, byte [g_char_num + 1]
	mov [g_int_b], r8d
	ret
