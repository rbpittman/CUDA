all: parallelThreads parallelBlocks parallelISumThreads testWarpSize maxDivergence

parallelThreads: parallelThreads.o
	g++ parallelThreads.o -lcudart -o $@

parallelThreads.o: parallelThreads.cu
	nvcc -c parallelThreads.cu

parallelBlocks: parallelBlocks.o
	g++ parallelBlocks.o -lcudart -o $@

parallelBlocks.o: parallelBlocks.cu
	nvcc -c parallelBlocks.cu



parallelISumThreads: parallelISumThreads.o
	g++ parallelISumThreads.o -lcudart -o $@

parallelISumThreads.o: parallelISumThreads.cu
	nvcc -c parallelISumThreads.cu


testWarpSize: testWarpSize.o
	g++ testWarpSize.o -lcudart -o $@

testWarpSize.o: testWarpSize.cu
	nvcc -c testWarpSize.cu


maxDivergence: maxDivergence.o
	g++ maxDivergence.o -lcudart -o $@

maxDivergence.o: maxDivergence.cu
	nvcc -c maxDivergence.cu
