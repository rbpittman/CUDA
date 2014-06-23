#include "constants.h"

#include <iostream>
#include <stdlib.h>

using namespace std;

//Post: modifies seed to be the next seed. 
//      Returns a pseudo-random number. 
inline int nextRandom(int & seed) {
  seed = NEXT_RANDOM(seed);
  return(seed);
}

//Pre: size > 0. 
//Post: Returns a square matrix with dimensions sizeXsize. 
//      All elements are random.
inline int * getMatrix(int size) {
  int * matrix = new int[size * size];
  int seed = RANDOM_SEED;
  for(int i = 0; i < size * size; i++) {
    matrix[i] = nextRandom(seed);
  }
  // int ** matrix = new int*[size];
  // for(int row = 0; row < size; row++) {
  //   matrix[row] = new int[size];
  //   for(int col = 0; col < size; col++) {
  //     matrix[row][col] = nextRandom(seed);
  //   }
  // }
  return(matrix);
}

// void printMatrix(int ** mat, int size) {
//   for(int row = 0; row < size; row++) {
//     for(int col = 0; col < size; col++) {
//       std::cout << mat[row][col];
//       if(col != size - 1) std::cout << ' ';
//     }
//     std::cout << std::endl;
//   }
// }

void printMatrix(int * mat, int size) {
  int i = 0;
  for(int row = 0; row < size; row++) {
    for(int col = 0; col < size; col++) {
      std::cout << mat[i];
      if(col != size - 1) std::cout << ' ';
      i++;
    }
    std::cout << std::endl;
  }
}

//Pre: This function must be called with at least matSize^2 threads.
__global__ void square(int * mat, int * result, int matSize) {
  int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if(idx < matSize * matSize) {
    int rowItemIndex = idx - (idx % matSize);
    int colItemIndex = idx % matSize;
    int total = 0;
    for(int i = 0; i < matSize; i++) {
      total += (mat[rowItemIndex]) * (mat[colItemIndex]);
      rowItemIndex++;
      colItemIndex += matSize;
    }
    result[idx] = total;
  }
}

inline int * squareMatrix(int * mat, int matrixSize) {
  size_t byteSize = matrixSize * matrixSize * sizeof(int);
  
  //Allocate device space for original matrix:
  int * gpuMatrix;
  cudaMalloc(&gpuMatrix, byteSize);
  //Copy the parameter mat matrix to device matrix:
  cudaMemcpy(gpuMatrix, mat, byteSize, cudaMemcpyHostToDevice);
  
  //Allocate result space on device:
  int * gpuResult;
  cudaMalloc(&gpuResult, byteSize);
  
  int numBlocks = ((matrixSize * matrixSize) / 512) + 1;
  //Run computation:
  square<<<numBlocks, 512>>>(gpuMatrix, gpuResult, matrixSize);

  //Allocate space for result:
  int * hostResult;
  hostResult = (int*) malloc(byteSize);
  //Copy gpu result to host:
  cudaMemcpy(hostResult, gpuResult, byteSize, cudaMemcpyDeviceToHost);
  
  //Cleanup:
  cudaFree(gpuMatrix);
  cudaFree(gpuResult);
  return(hostResult);
}

int main(int argc, char ** argv) {
  int matrixSize = DEFAULT_N;
  if(argc == 2) {
    matrixSize = atoi(argv[1]);
  } else {
    //std::cerr << "INVALID ARG SPECS" << std::endl;
  }
  //ASSERT: matrixSize is defined. 
  int * matrix = getMatrix(matrixSize);
  
  //Print random matrix:
  //printMatrix(matrix, matrixSize);
  
  int * hostResult = squareMatrix(matrix, matrixSize);
  
  //Print Result:
  //printMatrix(hostResult, matrixSize);
  
  delete[] matrix;
  delete[] hostResult;
  return(0);
}
