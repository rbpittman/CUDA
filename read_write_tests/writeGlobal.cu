#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

//#define DEBUG

using namespace std;


//Writes maximizing warp writing efficiency.
// __global__ void fastWriteGlobal(int * array, int size) {
//   int iSum = 0;
//   for(int i = 0; i < size; i++) {
//     iSum += i;
//   }
//   array[0] = iSum;  
// }

// __global__ void fastWriteReg(int * array, int size) {
//   int iSum = 0;
//   for(int i = 0; i < size; i++) {
//     iSum += i;
    
//   }
//   array[0] = iSum;
// }

__global__ void fastWriteGlobal(int * array, int size, int * forceRun) {
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  //__syncthreads();
  array[idx] = idx;
  //__syncthreads();
  forceRun[0] = idx;
}

__global__ void fastWriteReg(int * array, int size, int * forceRun) {
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  forceRun[0] = idx;
}

double convertToTime(boost::posix_time::ptime t1, boost::posix_time::ptime t2) {
  boost::posix_time::time_duration duration = t2 - t1;
  unsigned long micro = duration.total_microseconds();
  return(micro / 1000000.);
}

#ifdef DEBUG
//Pre: devArray and devForceRun are ptrs to device memory. 
void test(int * devArray, int size, int * devForceRun, bool isReg) {
  cout << "======" << endl
       << " TEST " << endl
       << "======" << endl;
  int random = 9827123;
  int * hostForceRun = new int(random);//Some random number
  cudaMemcpy(hostForceRun, devForceRun,  sizeof(int), cudaMemcpyDeviceToHost);
  int forceRun = (*hostForceRun);
  if(forceRun == random) {
    cout << "Debug error: devForceRun value was unchanged" << endl;
  } else if(!((0 <= forceRun) && (forceRun < size))) {
    cout << "Debug error: devForceRun value changed, but to an invalid idx value: "
	 << forceRun << endl;
  } else {
    cout << "PASS: forceRun was " << forceRun << endl;
  }
  
  int * hostArray = new int[size];
  cudaMemcpy(hostArray, devArray, size * sizeof(int), cudaMemcpyDeviceToHost);
  
  bool pass = true;
  for(int i = 0; i < size; i++) {
    if(isReg) {
      if(hostArray[i] != 0) {
 	// cout << "Debug error: Expected " << 0 << " but got " << hostArray[i] << endl;
	pass = false;
      }
    } else {
      if(hostArray[i] != i) {
	// cout << "Debug error: Expected " << i << " but got " << hostArray[i] << endl;
	pass = false;
      }
    }
  }
  
  if(pass) {
    cout << "PASS: hostArray was correct" << endl;
  } else {
    cout << "Debug error" << endl;
  }
}
#endif

#define NUM_THREADS 512
#define HOW_BIG_TO_MAKE_IT 419430400
#define WHAT_TO_MAKE_IT 0

void resetMemory() {
  bool * wiper;
  cudaMalloc(&wiper, HOW_BIG_TO_MAKE_IT * sizeof(bool));
  cudaMemset(wiper, WHAT_TO_MAKE_IT, HOW_BIG_TO_MAKE_IT * sizeof(bool));
  cudaFree(wiper);
}

void timeReg(int numBlocks, int numThreads, int size) {
  //Device array to be set to [0, 1, 2, ... size-1]:
  int * devArray; //An array of ints
  cudaMalloc(&devArray, size * sizeof(int));
  
  //Device array that forces fastWriteReg to run.
  int * devForceRun;//Just an int
  cudaMalloc(&devForceRun, sizeof(int));

  boost::posix_time::ptime regT1(boost::posix_time::microsec_clock::local_time());
  fastWriteReg<<<numBlocks, numThreads>>>(devArray, size, devForceRun);
  cudaDeviceSynchronize();
  boost::posix_time::ptime regT2(boost::posix_time::microsec_clock::local_time());
  
  #ifdef DEBUG
  test(devArray, size, devForceRun, true);
  #endif
  cout << "RegTime: " << convertToTime(regT1, regT2) << endl;
  
  cudaFree(devForceRun);
  cudaFree(devArray);
  resetMemory();
}

void timeGlobal(int numBlocks, int numThreads, int size) {
  //Device array to be set to [0, 1, 2, ... size-1]:
  int * devArray; //An array of ints
  cudaMalloc(&devArray, size * sizeof(int));
  
  //Device array that forces fastWriteReg to run.
  int * devForceRun;//Just an int
  cudaMalloc(&devForceRun, sizeof(int));

  boost::posix_time::ptime globalT1(boost::posix_time::microsec_clock::local_time());
  fastWriteGlobal<<<numBlocks, numThreads>>>(devArray, size, devForceRun);
  cudaDeviceSynchronize();
  boost::posix_time::ptime globalT2(boost::posix_time::microsec_clock::local_time());
#ifdef DEBUG
  test(devArray, size, devForceRun, false);
#endif
  cout << "GlobalTime: " << convertToTime(globalT1, globalT2) << endl;

  cudaFree(devForceRun);
  cudaFree(devArray);
  resetMemory();
}

int main(int argc, char ** argv) {
  if(argc != 2) {
    cout << "Expected number of threads to run\n";
  } else {
    resetMemory();
    int size = atoi(argv[1]);
    if(size % NUM_THREADS != 0) {
      cout << "Error: size must be divisible by " << NUM_THREADS << endl;
    } else {    
      int numThreads = NUM_THREADS;
      int numBlocks  = (size / numThreads);
      int max = (2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2);
      if(numBlocks >= max) {
	cout << "numBlocks was to big: " << numBlocks << endl;
      } else {
	timeGlobal(numBlocks, numThreads, size);
	timeReg(numBlocks, numThreads, size);
	timeGlobal(numBlocks, numThreads, size);
	timeReg(numBlocks, numThreads, size);
      }
    }
  }
  return(0);
}
