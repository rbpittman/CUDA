#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

#define START 0
#define END   1000
#define STEPS 10000

//#define SIZE 10000

double runBench(int size) {
  int * devMem;
  cudaMalloc(&devMem, size * sizeof(int));
  
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  
  for(int i = 0; i < STEPS; i++) {
    cudaMemset(devMem, 0, size * sizeof(int));
  }
  
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  double sec = (micro / 1000000.);
  return(sec);
}

int main() {
  for(int i = START; i < END + 1; i++) {
    runBench(i);
  }
  return(0);
}
