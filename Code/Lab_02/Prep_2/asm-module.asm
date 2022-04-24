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

	extern g_int_arr, g_int_suma, g_int_licha
	extern g_long_num, g_int_ones
	extern g_char_str, g_int_length, g_int_spaces

;***************************************************************************

    section .text


	global sum_int_arr
sum_int_arr:
	mov r8d, 0
	mov r9, 0

.loopbeg:
	add r8d, [g_int_arr + 4 * r9]

	inc r9
	cmp r9, 10
	jne .loopbeg

	mov [g_int_suma], r8d
	ret


	global negate_int_arr
negate_int_arr:
	mov r9, 0

.loopbeg:
	mov r8d, [g_int_arr + 4 * r9]
	not r8d
	inc r8d
	mov [g_int_arr + 4 * r9], r8d

	inc r9
	cmp r9, 10
	jne .loopbeg

	ret


	global count_odd
count_odd:
	mov r9, 0
	mov r8d, 0

.loopbeg:
	mov r10d, 1
	and r10d, [g_int_arr + 4 * r9]
	cmp r10d, 0
	je .next
	inc r8d

.next:
	inc r9
	cmp r9, 10
	jne .loopbeg

	mov [g_int_licha], r8d
	ret


	global count_ones
count_ones:
	mov r8, [g_long_num]
	mov r10d, 0

.loopbeg:
	mov r9, 1
	and r9, r8
	cmp r9, 0
	je .next
	inc r10d

.next:
	shr r8, 1
	cmp r8, 0
	jne .loopbeg

	mov [g_int_ones], r10d
	ret


	global str_length
str_length:
	mov r8d, -1

.loopbeg:
	inc r8d
	cmp [g_char_str + r8d], byte 0
	jne .loopbeg

	mov [g_int_length], r8d
	ret


	global count_spaces
count_spaces:
	mov r8d, 0
	mov r9, -1

.loopbeg:
	inc r9
	cmp [g_char_str + r9], byte ' '
	jne .next
	inc r8d
.next:
	cmp [g_char_str + r9], byte 0
	jne .loopbeg

	mov [g_int_spaces], r8d
	ret
