#include <iostream>
#include <stdlib.h>

#include "constants.h"

// Thread block size
#define BLOCK_SIZE 16

using namespace std;

// Device multiplication function called by Mul()
// Compute C = A * B
//wA is the width of A
//wB is the width of B
__global__ void Muld(int* A, int* B, int wA, int wB, int* C) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;
  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;
  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;
  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;
  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;
  // The element of the block sub-matrix that is computed
  // by the thread
  int Csub = 0;
  for (int a = aBegin, b = bBegin;
       a <= aEnd; 
       a += aStep, b += bStep) {
    // Shared memory for the sub-matrix of A
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    // Shared memory for the sub-matrix of B
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
    // Load the matrices from global memory to shared memory;
    // each thread loads one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];
    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    for (int k = 0; k < BLOCK_SIZE; ++k)  // Loop over all the
      // compute the block
      Csub += As[ty][k] * Bs[k][tx];
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write the block sub-matrix to global memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

// Host multiplication function
// Compute C = A * B
//hA is the height of A
//wA is the width of A
//wB is the width of B
void Mul(const int* A, const int* B, int hA, int wA, int wB,
	 int* C) {
  int size;
  // Load A and B to the device
  int* Ad;
  size = hA * wA * sizeof(int);
  cudaMalloc((void**)&Ad, size);
  cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);

  int* Bd;
  size = wA * wB * sizeof(int);
  cudaMalloc((void**)&Bd, size);
  cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
  // Allocate C on the device
  int* Cd;
  size = hA * wB * sizeof(int);
  cudaMalloc((void**)&Cd, size);
  // Compute the execution configuration assuming
  // the matrix dimensions are multiples of BLOCK_SIZE
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);
  // Launch the device computation
  Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
  // Read C from the device
  cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);
}

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
  return(matrix);
}

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
  
  size_t byteSize = matrixSize * matrixSize * sizeof(int);
  int * result = (int*)  malloc(byteSize);//= new int[matrixSize * matrixSize];
  
  
  //Run computation:
  Mul(matrix, matrix, matrixSize, matrixSize, matrixSize, result);
  
  //Print Result:
  //printMatrix(result, matrixSize);
  
  
  delete[] result;
  delete[] matrix;
  return(0);
}
