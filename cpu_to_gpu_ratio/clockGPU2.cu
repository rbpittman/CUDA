#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

#include <stdio.h>

using namespace std;

//#define STEPS 2147483647
#define STEPS 100000000

//The approximate minimum of the amount of time the timers pick up when
//there is no code in the kernel call. 
#define EXTRA_TIME 0.0001

// __global__ void runEmpty(int * count, int steps) {
//   for(int j = 0; j < steps; j++) {
//     for(int i = 0; i < steps; i++) {
//     }
//   }
//   count[0] = 0;
// }

__global__ void runGlobals(int * count) {
  for(int i = 0; i < STEPS; i++) {
    count[0] += i;
  }
}

__global__ void runSteps(int * count) {
  int c = 0;
  for(int i = 0; i < STEPS; i++) {
    c += i;
  }
  count[0] = c;
}

int main(int argc, char ** argv) {
  // int steps = STEPS;
  // if(argc == 1234) steps = 1;
  int * count;
  cudaMalloc(&count, sizeof(int));
  cudaMemset(count, 0, sizeof(int));
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  //runEmpty<<<1,1>>>(count, steps);
  runSteps<<<1,1>>>(count);
  // runGlobals<<<1,1>>>(count);
  cudaDeviceSynchronize();
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  
  int * hostCount = new int(123);
  cudaMemcpy(hostCount, count, sizeof(int), cudaMemcpyDeviceToHost);
  cout << "Count: " << *hostCount << endl;
  
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  double sec = (micro / 1000000.) - EXTRA_TIME;
  cout << sec << endl;
  return(0);
}
