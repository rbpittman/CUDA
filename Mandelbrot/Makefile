all: mand gpuMand

mand: mand.o 
	g++ mand.o -o mand

mand.o: mand.cc constants.h
	g++ -c mand.cc


# bench: bench.cc
# 	g++ bench.cc -o $@

gpuMand: gpuMand.o
	g++ gpuMand.o -lcudart -o gpuMand

gpuMand.o: gpuMand.cu constants.h
	nvcc -c gpuMand.cu

clean:
	rm *.o mand gpuMand 2> /dev/null

gpuMandLarge: gpuMandLarge.o
	g++ gpuMandLarge.o -lcudart -o gpuMandLarge

gpuMandLarge.o: gpuMandLarge.cu constants.h
	nvcc -c gpuMandLarge.cu


gpuMandSharedMem: gpuMandSharedMem.o
	g++ gpuMandSharedMem.o -lcudart -o gpuMandSharedMem

gpuMandSharedMem.o: gpuMandSharedMem.cu constants.h
	nvcc -c gpuMandSharedMem.cu

gpuMandSpeed: gpuMandSpeed.o
	g++ gpuMandSpeed.o -lcudart -o gpuMandSpeed

gpuMandSpeed.o: gpuMandSpeed.cu constants.h
	nvcc -c gpuMandSpeed.cu
