all: mult gpuMult example


mult: mult.o
	g++ mult.o -o mult

mult.o: mult.cc
	g++ -c mult.cc


gpuMult: gpuMult.o
	g++ gpuMult.o -lcudart -o gpuMult

gpuMult.o: gpuMult.cu
	nvcc -c gpuMult.cu


example: example.o
	g++ example.o -lcudart -o example

example.o: example.cu
	nvcc -c example.cu
