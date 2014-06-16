#include <iostream>
#include <stdlib.h>

#define DEFAULT_N 10 //A 10X10 matrix

#define RANDOM_SEED 37
#define A 19609
#define B 171
#define M 18307

//#define RAND_RANGE 101

#define NEXT_RANDOM(seed) (((seed * A) + B) % M)// % RAND_RANGE)

//Post: modifies seed to be the next seed. 
//      Returns a pseudo-random number. 
inline int nextRandom(int & seed) {
  seed = NEXT_RANDOM(seed);
  return(seed);
}

//Pre: size > 0. 
//Post: Returns a square matrix with dimensions sizeXsize. 
//      All elements are random.
inline int ** getMatrix(int size) {
  //setup random
  int seed = RANDOM_SEED;
  
  int ** matrix = new int*[size];
  for(int row = 0; row < size; row++) {
    matrix[row] = new int[size];
    for(int col = 0; col < size; col++) {
      matrix[row][col] = nextRandom(seed);
    }
  }
  return(matrix);
}

int main(int argc, char ** argv) {
  int matrixSize = DEFAULT_N;
  if(argc == 2) {
    matrixSize = atoi(argv[1]);
  } else {
    //std::cerr << "INVALID ARG SPECS" << std::endl;
  }
  //ASSERT: matrixSize is defined. 
  
  // int ** matrix = getMatrix(matrixSize);
  // for(int row = 0; row < matrixSize; row++) {
  //   for(int col = 0; col < matrixSize; col++) {
  //     std::cout << matrix[row][col] << ' ';
  //   }
  //   std::cout << std::endl;
  // }
  int ** squared = square(matrix, matrixSize);
  
  //DO I INCLUDE THE O(2n) operation of freeing 
  //memory??
  //delete[] matrix;
  return(0);
}
