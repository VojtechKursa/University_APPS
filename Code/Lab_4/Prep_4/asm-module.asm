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

    extern digit_counter

;***************************************************************************

    section .text

	global max_deviation
;RDI = t_array, RSI = t_len
max_deviation:
	mov eax, [rdi]
	mov rcx, 1
.1_begin:	; sum begin
	cmp rcx, rsi
	jge .1_end

	add eax, [rdi + rcx * 4]

	inc rcx
	jmp .1_begin
.1_end:		; sum end

	; avg
	cdq
	div rsi
	mov r9d, eax	; r9d = avg

	mov r11d, 0		; max deviation
	mov rcx, 0
.2_begin:
	cmp rcx, rsi
	jge .2_end

	mov r10d, [rdi + rcx * 4]
	sub r10d, r9d	; r10d = deviation

	cmp r10d, 0
	jge .2_skip1
	neg r10d
.2_skip1:

	cmp r10d, r11d
	cmovg eax, [rdi + rcx * 4]

	inc rcx
	jmp .2_begin
.2_end:
	ret


	global max_bits_one
;RDI = t_array, RSI = t_len
max_bits_one:
	mov rax, 0	; out
	mov rdx, 0	; max 1s
	mov rcx, 0	; loop counter
.1_begin:
	cmp rcx, rsi
	jge .1_end

	mov r9, [rdi + 8 * rcx]	; number holder
	mov r10, 1	; mask
	mov r11, 0	; curr 1s counter
.2_begin:
	cmp r10, 0
	je .2_end

	test r9, r10
	je .2_skip1
	inc r11
.2_skip1:

	shl r10, 1
	jmp .2_begin
.2_end:

	cmp r11, rdx
	cmovg rax, r9
	cmovg rdx, r11

	inc rcx
	jmp .1_begin
.1_end:
	ret


	global to_num
;RDI = t_str
to_num:
	mov rcx, 0
.1_begin:
	cmp byte [rdi + rcx], 0
	je .1_end

	inc rcx
	jmp .1_begin
.1_end:
	dec rcx

	mov rsi, 0	; char holder
	mov r8d, 1	; order
	mov r9d, 0	; result/sum
	mov r10d, 10	; order multiplier
.2_begin:
	cmp rcx, 0
	jl .2_end

	mov sil, [rdi + rcx]	; char holder
	cmp sil, '-'
	je .2_neg_beg

	cmp sil, '0'
	jb .2_neg_end
	cmp sil, '9'
	ja .2_neg_end

	sub sil, '0'
	movzx eax, sil
	mul r8d
	add r9d, eax

	mov eax, r8d
	mul r10d
	mov r8d, eax

	jmp .2_neg_end
.2_neg_beg:
	neg r9d
.2_neg_end:

	dec rcx
	jmp .2_begin
.2_end:

	mov eax, r9d
	ret


	global in_circle
;RDI = t_x, RSI = t_y, RDX = t_R
in_circle:
	mov ecx, edx

	mov eax, edi
	imul edi
	mov r9d, eax
	shl rdx, 32
	or r9, rdx		; r9 = x^2

	mov eax, esi
	imul esi
	mov r10d, eax
	shl rdx, 32
	or r10, rdx		; r10 = y^2

	mov eax, ecx
	imul ecx
	mov r11d, eax
	shl rdx, 32
	or r11, rdx		; r11 = R^2

	add r9, r10		; r9 = x^2 + y^2

	mov eax, 0
	cmp r9, r11
	jg .skip
	mov eax, 1
.skip:
	ret


	global digit
;RDI = t_str
digit:
	mov rcx, 0
.1_begin:
	mov al, [rdi + rcx]
	cmp al, 0
	je .1_end

	cmp al, '0'
	jb .1_skip1
	cmp al, '9'
	ja .1_skip1

	sub al, '0'
	movzx rax, al
	inc byte [digit_counter + rax]

.1_skip1:

	inc rcx
	jmp .1_begin
.1_end:

	mov al, 0	; out
	mov rcx, 0	; loop counter
	mov dl, 0	; max occurence
.2_begin:
	cmp rcx, 10
	jge .2_end

	cmp [digit_counter + rcx], dl
	jbe .2_skip1
	mov al, cl
	mov dl, [digit_counter + rcx]
.2_skip1:

	inc rcx
	jmp .2_begin
.2_end:

	add al, '0'
	ret

