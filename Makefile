STYLE="{BasedOnStyle: llvm, IndentWidth: 8}"
FORMATTER=clang-format-3.8

all: fft.c pgm.h
	gcc -g fft.c -o fft -lOpenCL -lm

format:
	${FORMATTER} -style=${STYLE} -i fft.c
	${FORMATTER} -style=${STYLE} -i pgm.h
