all: clockGPU clockGPU2 clockCPU clockCPU2 memcpyTime mallocTime

clockGPU: clockGPU.o
	g++ clockGPU.o -lcudart -o clockGPU

clockGPU.o: clockGPU.cu
	nvcc -c clockGPU.cu

clockGPU2: clockGPU2.o
	g++ clockGPU2.o -lcudart -o clockGPU2

clockGPU2.o: clockGPU2.cu
	nvcc -c clockGPU2.cu

memcpyTime: memcpyTime.o
	g++ memcpyTime.o -lcudart -o memcpyTime

memcpyTime.o: memcpyTime.cu
	nvcc -c memcpyTime.cu

mallocTime: mallocTime.o
	g++ mallocTime.o -lcudart -o mallocTime

mallocTime.o: mallocTime.cu
	nvcc -c mallocTime.cu


clockCPU2: clockCPU2.o
	g++ clockCPU2.o -o $@

clockCPU2.o: clockCPU2.cc
	g++ -c clockCPU2.cc


clockCPU: clockCPU.o
	g++ clockCPU.o -o $@

clockCPU.o: clockCPU.cc
	g++ -c clockCPU.cc
