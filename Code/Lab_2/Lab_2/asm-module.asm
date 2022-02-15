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
    
    extern g_long, g_longsum, g_long2, g_long2sum
    extern g_int_arr, g_int_index_min, g_evensum

;***************************************************************************

    section .text
    

    global sum_long
sum_long:
	mov al, 0
	mov rcx, 0

.begin:
	cmp rcx, 8
	jae .end

	add al, [g_long + rcx]

	inc rcx
	jmp .begin
.end:

	mov [g_longsum], al
	ret


    global sum_long2
sum_long2:
	mov eax, 0
	mov rcx, 0
	mov edx, 0

.begin:
	cmp rcx, 8
	jae .end

	mov dl, [g_long2 + rcx]
	add eax, edx

	inc rcx
	jmp .begin
.end:

	mov [g_long2sum], eax
	ret


	global find_min
find_min:
	mov eax, [g_int_arr]	;min
	mov ecx, 1				;counter
	mov r9d, 0				;min_index

.begin:
	cmp rcx, 10
	jae .end

	cmp [g_int_arr + ecx * 4], eax
	jge .skip
	mov eax, [g_int_arr + ecx * 4]
	mov r9d, ecx

.skip:
	inc rcx
	jmp .begin
.end:

	mov [g_int_index_min], r9d
	ret


	global sum_even
sum_even:
	mov eax, 0	;sum
	mov ecx, 0	;counter
	mov r9d, 1	;mask

.begin:
	cmp ecx, 10
	jae .end

	test r9d, [g_int_arr + 4*ecx]
	jne .skip	;jnz
	add eax, [g_int_arr + 4*ecx]

.skip:
	inc ecx
	jmp .begin
.end:

	mov [g_evensum], eax
	ret
