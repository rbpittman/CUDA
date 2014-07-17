#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

//#define DEBUG

using namespace std;

#define NTH_PRIME 620
#define STEP 4

//Pre: There is only one block of threads. 
//     sum length is the same as the number of threads. 
//Post: Sets every element of sum to the nth prime as declared 
//      by NTH_PRIME. 
__global__ void prime(int * sum) {
  //int idx = threadIdx.x;
  int count = 0;
  int curr = 2;
  int total = 0;
  while(count < NTH_PRIME) {
    bool isPrime = true;
    for(int i = 2; (i < curr) && (isPrime); i++) {
      isPrime = curr % i != 0;
    }
    if(isPrime) {
      total += curr;
      count++;
    }
    curr++;
  }
  sum[0] = total;
}

//Pre: 0 <= size <= 512
double runTest(int size) {
  int * devSum;
  cudaMalloc(&devSum, sizeof(int));
  
  //int blocks = 1;
  int blocks = 2;
  int threads = size;
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  prime<<<blocks,threads>>>(devSum);
  cudaDeviceSynchronize();
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  int * hostSum = new int(123456);
  cudaMemcpy(hostSum, devSum, sizeof(int), cudaMemcpyDeviceToHost);
  
  #ifdef DEBUG
  cout << "Host sum: " << (*hostSum) << endl;
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
  cout << "Parallel thread benchmark\nTrials: " << TRIALS << endl;
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
  return(0);
}
