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

;***************************************************************************

    section .text

    global move_int
;RDI = t_int_arr, RSI = t_out_arr, RDX = t_n
move_int:
	mov rax, 0		; out
	mov r10, 0		; loop counter
.begin:
	cmp r10, rdx
	jge .end

	cmp dword [rdi + r10 * 4], 0
	jl .skip
	mov r11d, [rdi + r10 * 4]
	mov [rsi + rax * 4], r11d
	inc rax

.skip:
	inc r10
	jmp .begin
.end:
	ret


    global multiple_24
;RDI = t_arr, RSI = t_n
multiple_24:
	mov rax, 0		; out
	mov r10, 0		; loop counter
.begin:
	cmp r10, rsi
	jge .end

	test dword [rdi + r10 * 4], 1
	jne .skip
	inc rax

.skip:
	inc r10
	jmp .begin
.end:
	ret


	global negative_max
;RDI = t_arr, RSI = t_n
negative_max:
	mov rax, 0x8000000000000000		; out
	mov r10, 0		; loop counter
.begin:
	cmp r10, rsi
	jge .end

	cmp qword [rdi + r10 * 8], 0
	jge .skip
	cmp rax, [rdi + r10 * 8]
	jge .skip
	mov rax, [rdi + r10 * 8]

.skip:
	inc r10
	jmp .begin
.end:
	ret


	global biggest_byte
;RDI = t_num
biggest_byte:
	mov al, 0	; out
	mov r10, 0	; loop counter

.begin:
	cmp r10, 8
	jge .end

	shr rdi, 8
	cmp al, dil
	jge .skip
	mov al, dil

.skip:
	inc r10
	jmp .begin
.end:
	ret


	global highest_bit
;RDI = t_num
highest_bit:
	mov eax, 0						; out + counter
	mov r10, 0x8000000000000000		; mask

.begin:
	cmp r10, 0
	je .end

	test r10, rdi
	jne .end

	shr r10, 1
	inc eax
	jmp .begin
.end:
	ret


	global replace_digit
;RDI = t_str, RSI = t_replace
replace_digit:
	mov eax, 0	; out
	mov r10, 0 	; index counter

.begin:
	cmp byte [rdi + r10], 0
	je .end

	cmp byte [rdi + r10], '0'
	jb .skip
	cmp byte [rdi + r10], '9'
	ja .skip

	mov [rdi + r10], sil
	inc eax

.skip:
	inc r10
	jmp .begin
.end:
	ret
