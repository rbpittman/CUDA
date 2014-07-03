#include <iostream>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

#define PRIMES 500

void runSteps(int * primes) {
  int count = 0;
  int currCheck = 2;
  while(count < PRIMES) {
    bool isPrime = true;
    for(int i = 2; i < currCheck - 1; i++) {
      if(currCheck % i == 0) {
	isPrime = false;
      }
    }
    if(isPrime) {
      primes[count] = currCheck;
      count++;
    }
    currCheck++;
  }
}

void printPrimes(int * primes) {
  for(int i = 0; i < PRIMES; i++) {
    cout << primes[i] << ' ';
  }
  cout << "\b\n";
}

int main() {
  int * primes = new int[PRIMES];
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  runSteps(primes);
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  // printPrimes(primes);
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  double sec = micro / 1000000.;
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
