#include <iostream>
#include <stdlib.h>

#define DEFAULT_N 10 //A 10X10 matrix

#define RANDOM_SEED 37
#define A 19609
#define B 171
#define M 3301

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

//Pre: mat is a 2D array with dimensions size X size. 
//Post: Returns mat * mat.
//Complexity: O(n^3), where n = size. 
int ** square(int ** mat, int size) {
  int ** result = new int*[size];
  //Iterate over each spot in the result matrix:
  for(int row = 0; row < size; row++) {
    //Init the new row of the result:
    result[row] = new int[size];
    for(int col = 0; col < size; col++) {
      //Iterate over the row of first and column of second.
      int total = 0;
      for(int i = 0; i < size; i++) {
	total += (mat[row][i] * mat[i][col]);
      }
      result[row][col] = total;
    }
  }
  return(result);
}

void printMatrix(int ** mat, int size) {
  for(int row = 0; row < size; row++) {
    for(int col = 0; col < size; col++) {
      std::cout << mat[row][col];
      if(col != size - 1) std::cout << ' ';
    }
    std::cout << std::endl;
  }
}

int main(int argc, char ** argv) {
  int matrixSize = DEFAULT_N;
  if(argc == 2) {
    matrixSize = atoi(argv[1]);
  } else {
    //std::cerr << "INVALID ARG SPECS" << std::endl;
  }
  //ASSERT: matrixSize is defined. 
  int ** matrix = getMatrix(matrixSize);
  
  //Print random matrix:
  //printMatrix(matrix, matrixSize);
  
  //Run computation:
  int ** squared = square(matrix, matrixSize);
  
  //Print Result:
  //printMatrix(squared, matrixSize);
  
  //DO I INCLUDE THE O(2n) operation of freeing 
  //memory??
  //delete[] matrix;
  return(0);
}
