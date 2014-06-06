all: mand bench gpuMand

mand: mand.o 
	g++ mand.o -o mand

mand.o: mand.cc constants.h
	g++ -c mand.cc


bench: bench.cc
	g++ bench.cc -o $@

gpuMand: gpuMand.o
	g++ gpuMand.o -lcudart -o gpuMand

gpuMand.o: gpuMand.cu constants.h
	nvcc -c gpuMand.cu

clean:
	rm *.o mand gpuMand 2> /dev/null
