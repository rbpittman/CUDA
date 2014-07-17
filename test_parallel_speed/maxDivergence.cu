#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

#define WARP_SIZE 32
#define DEBUG false

#define LOOPS 1000000
#define TEST_ONE_THREAD false

#define TRIALS 2

//Pre: number of threads % WARP_SIZE == 0.
__global__ void diverge(int * force) {
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int switcher = idx % WARP_SIZE;
  switch(switcher) {
  case 0:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 0;
    break;
  case 1:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 1;
    break;
  case 2:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 2;
    break;
  case 3:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 3;
    break;
  case 4:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 4;
    break;
  case 5:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 5;
    break;
  case 6:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 6;
    break;
  case 7:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 7;
    break;
  case 8:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 8;
    break;
  case 9:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 9;
    break;
  case 10:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 10;
    break;
  case 11:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 11;
    break;
  case 12:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 12;
    break;
  case 13:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 13;
    break;
  case 14:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 14;
    break;
  case 15:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 15;
    break;
  case 16:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 16;
    break;
  case 17:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 17;
    break;
  case 18:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 18;
    break;
  case 19:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 19;
    break;
  case 20:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 20;
    break;
  case 21:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 21;
    break;
  case 22:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 22;
    break;
  case 23:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 23;
    break;
  case 24:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 24;
    break;
  case 25:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 25;
    break;
  case 26:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 26;
    break;
  case 27:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 27;
    break;
  case 28:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 28;
    break;
  case 29:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 29;
    break;
  case 30:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 30;
    break;
  case 31:
    for(int i = idx; i < idx + LOOPS; i++) *force += i;
    *force += 31;
    break;
  }
}

//Pre: Runs 2*size threads in 2 blocks. 
double runTest(int size) {
  int * devForce;
  cudaMalloc(&devForce, sizeof(int));
  cudaMemset(&devForce, 0, sizeof(int));

  int blocks = 2;
  int threads = size;
  
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  diverge<<<blocks,threads>>>(devForce);
  cudaDeviceSynchronize();
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  
  if(DEBUG) {
    int * hostForce = new int(123);
    cudaMemcpy(hostForce, devForce, sizeof(int), cudaMemcpyDeviceToHost);
    cout << (*hostForce) << endl;
  }
  
  cudaFree(devForce);
  
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  return (micro / 1000000.);
}

int main() {
  if(TEST_ONE_THREAD) {
    cout << runTest(1) << endl;
  } else {
    cout << "Testing times for maximum warp divergence" << endl;
    cout << "Trials: " << TRIALS << endl;
    for(int size = 0; size <= 512; size += 4) {
      cout << 2 * size << endl;
      for(int i = 0; i < TRIALS; i++) {
	cout << "0m" << runTest(size);
	if(i != TRIALS - 1) {
	  cout << " ";
	}
      }
      cout << endl;
    }
  }
  return(0);
}
