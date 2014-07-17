#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

//#define DEBUG

using namespace std;

// #define NTH_SUM 13749680
#define NTH_SUM 13000000
#define STEP 4

#define TEST_ONE_THREAD false


//Pre: There is only one block of threads. 
//     sum length is the same as the number of threads. 
//Post: Sets every element of sum to the nth sum as declared 
//      by NTH_SUM. 
//NOTE: This kernel is designed so that all threads will always execute SIMT
//      fashion. i.e. all instructions will always be the same, except each thread
//      will have different data (the idx value)
__global__ void sum(int * sum) {
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int total = 0;
  for(int i = 0; i < NTH_SUM; i++) {
    total += i;
  }
  sum[idx] = total;
}

//Pre: 0 <= size <= 512
double runTest(int size) {
  int * devSum;
  cudaMalloc(&devSum, sizeof(int) * size);
  
  int blocks = 2;//To go up to 1024
  if(TEST_ONE_THREAD) {
    blocks = 1; //Then only use 1 block, so we can have 1 line of execution.
  }
  int threads = size;
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  sum<<<blocks,threads>>>(devSum);
  cudaDeviceSynchronize();
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  int * hostSum = new int[size];
  cudaMemcpy(hostSum, devSum, sizeof(int) * size, cudaMemcpyDeviceToHost);
  
  #ifdef DEBUG
  cout << "Host sum:\n{";
  for(int i = 0; i < size; i++) {
    cout << hostSum[i];
    if(i != size - 1) cout << ", ";
  }
  cout << '}' << endl;
  #endif
  
  delete[] hostSum;
  cudaFree(devSum);
  
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  return (micro / 1000000.);
}

#define TRIALS 2

int main() {
  runTest(10);//Warm up
  if(TEST_ONE_THREAD) {
    cout << runTest(1) << endl;
  } else {
    cout << "Parallel thread benchmarking iSums\nTrials: " << TRIALS << endl;
    for(int size = 0; size <= 512; size += STEP) {
      cout << 2 * size << endl;
      // cout << size << endl;
      for(int i = 0; i < TRIALS; i++) {
	cout << "0m" << runTest(size);
	if(i != TRIALS - 1) {
	  cout << ' ';
	}
      }
      cout << endl;
    }
  }
  return(0);
}
