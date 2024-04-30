build: spindafy.cu kernel.cu kernel.h
	nvcc kernel.cu spindafy.cu -o spinda

run: build
	./spinda