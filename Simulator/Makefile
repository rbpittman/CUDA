simulator: simulator.o
	g++ simulator.o -lcudart -o simulator

simulator.o: simulator.cu constants.h
	nvcc -c simulator.cu

cpuSimulator: cpuSimulator.o
	g++ cpuSimulator.o -lcudart -o cpuSimulator

cpuSimulator.o: cpuSimulator.cc
	g++ -c cpuSimulator.cc

primes: generatePrimes.o
	g++ generatePrimes.o -o generatePrimes

generatePrimes.o : generatePrimes.cc
	g++ -c generatePrimes.cc

# -I/home/randall/boost/boost_1_55_0

# timer.o: timer.cc timer.h
# 	g++ -c timer.cc -std=c++0x

