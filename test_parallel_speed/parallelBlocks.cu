#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

//#define DEBUG

using namespace std;

#define NTH_PRIME 400
#define NUM_THREADS 32

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
  __syncthreads();
  sum[0] = total;
}

//Pre: 0 <= size <= 512
//     size % NUM_THREADS == 0
double runTest(int size) {
  int * devSum;
  cudaMalloc(&devSum, sizeof(int));
  
  int blocks = size / NUM_THREADS;
  int threads = NUM_THREADS;
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

int main() {
  runTest(10);//Warm up
  for(int theValueThatTellsHowManyThreadsToRun = 0; 
      theValueThatTellsHowManyThreadsToRun <= 512; 
      theValueThatTellsHowManyThreadsToRun += NUM_THREADS) {
    cout << theValueThatTellsHowManyThreadsToRun << endl 
	 << "0m" << runTest(theValueThatTellsHowManyThreadsToRun) << ' '
	 << "0m" << runTest(theValueThatTellsHowManyThreadsToRun) << endl;
  }
  return(0);
}
