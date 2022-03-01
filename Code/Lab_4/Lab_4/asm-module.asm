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


	global even_nibbles
;RDI = arr, RSI = n
even_nibbles:
	mov eax, 0
	mov r8, 0	; loop counter
	mov r9d, 2
.1_begin:
	cmp r8, rsi
	jge .1_end

	mov ecx, [rdi + r8 * 4]
.2_begin:
	cmp ecx, 0
	je .return_b

	jmp .return_e
.return_b:
	mov eax, [rdi + r8 * 4]
	ret
.return_e:

	mov eax, 0xF
	and eax, ecx

	cdq
	idiv r9d

	cmp edx, 0
	jne .2_end

	shr ecx, 4
	jmp .2_begin
.2_end:

	inc r8
	jmp .1_begin
.1_end:

	mov eax, -1
	ret


    global under_avg
;RDI = arr, RSI = n
under_avg:
	mov rax, 0
	mov rcx, 0	; loop counter
.1_begin:
	cmp rcx, rsi
	jge .1_end

	add rax, [rdi + rcx * 8]

	inc rcx
	jmp .1_begin
.1_end:

	cqo
	idiv rsi

	mov r8, rax
	mov rcx, 0
	mov rax, 0
.2_begin:
	cmp rcx, rsi
	jge .2_end

	cmp [rdi + rcx * 8], r8
	jge .2_skip
	inc eax
.2_skip:

	inc rcx
	jmp .2_begin
.2_end:
	ret


    global most_common
;RDI = arr, RSI = n
most_common:
	mov eax, 0		; most common num
	mov edx, 0		; most common count
	mov r11d, 50	; curr searched number
.1_begin:
	cmp r11d, 500
	jg .1_end

	mov rcx, 0	; loop counter
	mov r10d, 0	; curr count

.2_begin:
	cmp rcx, rsi
	jge .2_end

	cmp r11d, [rdi + 4 * rcx]
	jne .2_skip1
	inc r10d
.2_skip1:

	inc rcx
	jmp .2_begin
.2_end:

	cmp r10d, edx
	cmova edx, r10d
	cmova eax, r11d

	add r11d, 50
	jmp .1_begin
.1_end:
	ret


	global shorter_vector
;RDI = v1, RSI = v2
shorter_vector:
	mov rcx, 0
	mov r8, 0	; v1 len
	mov r9, 0	; v2 len

.begin:
	cmp rcx, 3
	jge .end

	mov eax, [rdi + 4 * rcx]
	imul eax
	shl rdx, 32
	or rax, rdx
	add r8, rax

	mov eax, [rsi + 4 * rcx]
	imul eax
	shl rdx, 32
	or rax, rdx
	add r9, rax

	inc rcx
	jmp .begin
.end:

	cmp r8, r9
	jle .ret_1
	mov eax, 2
	ret
.ret_1:
	mov eax, 1
	ret
