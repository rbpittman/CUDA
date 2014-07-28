all: gpuMand mand

gpuMand: gpuMand.o
	g++ gpuMand.o -lcudart -o $@
gpuMand.o: gpuMand.cu constants.h Makefile
	nvcc -G0 -O0 -Xopencc -O0 -Xptxas -O0 -c gpuMand.cu

mand: mand.o
	g++ mand.o -o $@
mand.o: mand.cc
	g++ -c mand.cc
