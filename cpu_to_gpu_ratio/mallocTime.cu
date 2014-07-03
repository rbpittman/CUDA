#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

#define STEPS 10000
//#define SIZE 10000

double runBench(int size, int steps) {
  int * devMem;
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  for(int i = 0; i < steps; i++) {
    cudaMalloc(&devMem, size * sizeof(int));
    // cudaFree(devMem);
  }
  cudaDeviceSynchronize();
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  double sec = (micro / 1000000.);
  return(sec);
}

int main(int argc, char ** argv) {
  // if(argc != 2) {
  //   cout << "Invalid number of arguments" << endl;
  // } else {
  //===
  //Get rid of library init
  int * temp;
  cudaMalloc(&temp, sizeof(int));
  cudaFree(temp);
  //===
  //for(int size = 0; size < 1200000000; size += 10000000); //max 
  for(int size = 0; size <   120000000; size += 1000000) {
    cout << size << ": " << runBench(size, STEPS) << endl;
  }
  return(0);
}
