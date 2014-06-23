import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy, sys
from constants import *

#Post: Returns a 1D numpy array of random elements of length size^2. 
def getMatrix(size):
    return numpy.random.randint(0, M_VALUE, size * size)

def printMatrix(mat, size):
    i = 0
    for i in range(size * size):
        print mat[i],
        i += 1
        print "\b"

module = SourceModule("""
__global__ void Muld(int* A, int* B, int wA, int wB, int* C) {
  #define BLOCK_SIZE 16
  
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
""")

def squareMatrix(mat, matSize):
    # Allocate device space for original matrix:
    gpuMat = cuda.mem_alloc(hostMatrix.nbytes)
    # Copy the parameter mat matrix to device matrix:
    cuda.memcpy_htod(gpuMat, mat)
     
    byteSize = numpy.int32().nbytes * matSize * matSize
    # Allocate result space on device:
    gpuResult = cuda.mem_alloc(byteSize)
    
    # Run computation:
    square = module.get_function("Muld")
    square(gpuMat, gpuMat, numpy.int32(matSize), numpy.int32(matSize), gpuResult,
           grid=(matSize // 16, matSize // 16), block=(16, 16, 1))
    
    # Allocate space for result:
    hostResult = numpy.empty(matSize * matSize, dtype=numpy.int32)
    # Copy gpu result to host:
    cuda.memcpy_dtoh(hostResult, gpuResult)
    
    # Cleanup:
    gpuMat.free()
    gpuResult.free()
    return hostResult

if __name__ == "__main__":
    matSize = DEFAULT_N
    if len(sys.argv) == 2:
        matSize = int(sys.argv[1])
    # ASSERT: matSize is defined. 
    matrix = getMatrix(matSize)
    #Get host random matrix as a numpy array:
    hostMatrix = getMatrix(matSize)

    # Print random matrix:
    # print "RANDOM MATRIX"
    # printMatrix(matrix, matSize)
  
    hostResult = squareMatrix(matrix, matSize)
    
    # Print Result:
    # print "SQUARED"
    # printMatrix(hostResult, matSize)
