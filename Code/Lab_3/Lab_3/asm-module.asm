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
    
    global search_amax
;	RDI = arr, RSI = len
search_amax:
	mov rax, [rdi]	; out, curr_max
	mov rcx, 1		; loop counter

.begin:
	cmp rcx, rsi
	jge .end

	mov r10, [rdi + 8 * rcx]
	cmp r10, 0
	jge .skip1
	neg r10

.skip1:
	cmp r10, rax
	jle .skip2
	mov rax, r10

.skip2:
	inc rcx
	jmp .begin
.end:

	ret


	global search_mask
;	RDI = arr, RSI = len, RDX = bits
search_mask:
	mov eax, 0	; out

	mov rcx, 32
	sub rcx, rdx
	mov r10d, 0xFFFFFFFF	; mask
	shr r10d, cl

	mov rcx, 0	; loop counter

.begin:
	cmp rcx, rsi
	jge .end

	test [rdi + 4*rcx], r10d
	jne .skip
	inc eax

.skip:
	inc rcx
	jmp .begin
.end:

	ret


	global copy_nospace
;	RDI = str_in, RSI = str_out
copy_nospace:
	mov rcx, 0		; loop counter
	mov dl, 0		; buffer
	mov r11, 0		; out counter

.begin:
	cmp byte [rdi + rcx], 0
	je .end

	cmp byte [rdi + rcx], ' '
	je .skip
	mov dl, [rdi + rcx]
	mov [rsi + r11], dl
	inc r11

.skip:
	inc rcx
	jmp .begin
.end:

	mov byte [rsi + r11], 0
	ret


	global num_digits
;	RDI = str_in
num_digits:
	mov rax, 0		; out
	mov rcx, 0		; loop counter

.begin:
	cmp byte [rdi + rcx], 0
	je .end

	cmp byte [rdi + rcx], '0'
	jb .skip
	cmp byte [rdi + rcx], '9'
	ja .skip
	inc rax

.skip:
	inc rcx
	jmp .begin
.end:

	ret
