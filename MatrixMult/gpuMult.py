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
// Pre: This function must be called with at least matSize^2 threads.
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
""")

def squareMatrix(mat, matSize):
    # Allocate device space for original matrix:
    gpuMat = cuda.mem_alloc(hostMatrix.nbytes)
    # Copy the parameter mat matrix to device matrix:
    cuda.memcpy_htod(gpuMat, mat)
     
    byteSize = numpy.int32().nbytes * matSize * matSize
    # Allocate result space on device:
    gpuResult = cuda.mem_alloc(byteSize)
    
    numBlocks = ((matSize * matSize) // 512) + 1
    # Run computation:
    square = module.get_function("square")
    square(gpuMat, gpuResult, numpy.int32(matSize), grid=(numBlocks, 1), block=(512, 1, 1))
    
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
