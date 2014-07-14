all: parallelThreads parallelBlocks

parallelThreads: parallelThreads.o
	g++ parallelThreads.o -lcudart -o $@

parallelThreads.o: parallelThreads.cu
	nvcc -c parallelThreads.cu

parallelBlocks: parallelBlocks.o
	g++ parallelBlocks.o -lcudart -o $@

parallelBlocks.o: parallelBlocks.cu
	nvcc -c parallelBlocks.cu