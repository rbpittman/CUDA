#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

#define STEPS 10000

//#define SIZE 10000

int main(int argc, char ** argv) {
  if(argc != 2) {
    cout << "Invalid number of arguments" << endl;
  } else {
    int size = atoi(argv[1]);
    int * devMem;
    cudaMalloc(&devMem, size * sizeof(int));
  
    int * hostMem = new int[size];
  
    boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  
    for(int i = 0; i < STEPS; i++) {
      cudaMemcpy(hostMem, devMem, size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration = t2 - t1;
    long micro = duration.total_microseconds();
    double sec = (micro / 1000000.);
    cout << sec << endl;
  }
  return(0);
}
