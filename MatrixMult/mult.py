
import sys
from constants import *

# DEFAULT_N = 10 #A 10X10 matrix

# #WARNING: MAKE SURE FOR BENCHMARKS THAT THESE ARE THE SAME
# #AS IN THE CC FILES->
# RANDOM_SEED = 37
# A = 19609
# B = 171
# M = 3301
#RAND_RANGE = 101

#NEXT_RANDOM = lambda seed: (((seed * A) + B) % M)# % RAND_RANGE)

#Pre: size > 0. 
#Post: Returns a square matrix with dimensions sizeXsize. 
#      All elements are random.
def getMatrix(size):
    # setup random
    seed = RANDOM_SEED
    matrix = [[0] * size for i in range(size)]
    for row in range(size):
        for col in range(size):
            seed = ((seed * A) + B) % M
            matrix[row][col] = seed
    return matrix

#Pre: mat is a 2D array with dimensions size X size. 
#Post: Returns mat * mat.
#Complexity: O(n^3), where n = size. 
def square(mat, size):
    result = [[0] * size for i in range(size)]
    # Iterate over each spot in the result matrix:
    for row in range(size):
        for col in range(size):
            # Iterate over the row of first and column of second.
            total = 0
            for i in range(size):
                total += (mat[row][i] * mat[i][col])
            result[row][col] = total
    return result

def printMatrix(mat):
    for i in range(len(mat)):
        for j in range(len(mat)):
            print mat[i][j], 
        print '\b'

if __name__ == "__main__":
    matrixSize = DEFAULT_N
    if len(sys.argv) == 2:
        matrixSize = int(sys.argv[1])
    # ASSERT: matrixSize is defined. 
    matrix = getMatrix(matrixSize)
    # Print random matrix:
    printMatrix(matrix)
  
    # Run computation:
    squared = square(matrix, matrixSize)
    
    # Print Result:
    printMatrix(squared)
    
    # DO I INCLUDE THE O(2n) operation of freeing 
    # memory??
    # delete[] matrix;
