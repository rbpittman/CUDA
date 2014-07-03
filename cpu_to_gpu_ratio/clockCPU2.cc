#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

#define STEPS 1000000000

#define EXTRA_TIME 0.0001

void runSteps(int * count) {
  int c = 0;
  for(int i = 0; i < STEPS; i++) {
    c += i;
  }
  count[0] = c;
}

int main() {
  int * count = new int(0);
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  runSteps(count);
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  cout << "Count: " << *count << endl;
  
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  double sec = (micro / 1000000.) - EXTRA_TIME;
  cout << sec << endl;
  return(0);
}














// #include <iostream>
// #include "boost/date_time/posix_time/posix_time.hpp"

// using namespace std;

// #define STEPS 100000000

// __global__ void runSteps(int * steps) {
//   int total = 0;
//   for(int i = 0; i < *steps; i++) {
//     total += i;
//   }
//   steps = &total;
// }

// int main() {
//   //Get rid of library initialization:
//   //=========
//   int * temp;
//   cudaMalloc(&temp, sizeof(int));
//   cudaFree(temp);
//   //=========
  
//   boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
//   // int * list;
//   // cudaMalloc(&list, STEPS * sizeof(int));
//   int hostSteps = STEPS;
//   int * steps;
//   cudaMalloc(&steps, sizeof(int));
//   cudaMemcpy(steps, &hostSteps, sizeof(int), cudaMemcpyHostToDevice);
//   runSteps<<<1, 1>>>(steps);
//   cudaMemcpy(&hostSteps, steps, sizeof(int), cudaMemcpyDeviceToHost);
//   cout << hostSteps << endl;
//   cudaDeviceSynchronize();
//   boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
//   boost::posix_time::time_duration duration = t2 - t1;
//   long micro = duration.total_microseconds();
//   double sec = micro / 1000000.;
//   cout << sec << endl;
//   return(0);
// }
