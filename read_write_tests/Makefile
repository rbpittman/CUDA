all: writeGlobal readGlobal


writeGlobal: writeGlobal.o
	g++ writeGlobal.o -lcudart -o $@

writeGlobal.o: writeGlobal.cu
	nvcc -c writeGlobal.cu

readGlobal: readGlobal.o
	g++ readGlobal.o -lcudart -o $@

readGlobal.o: readGlobal.cu
	nvcc -c readGlobal.cu
