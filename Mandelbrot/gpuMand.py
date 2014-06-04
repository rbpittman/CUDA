import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
from constants import *
import time

module = SourceModule("""
#include "/home/administrator/CUDA/Mandelbrot/constants.h"

//Pre: x is defined. 
//Post: Converts x from an image array pixel index to 
//      the real part of the complex graph location that the
//      pixel represents. 
inline __device__ float pixelXToComplexReal(uint x) {
  return(((x / ((float) WINDOW_WIDTH)) * SIZE) + START_X - (SIZE / 2.f));
}

//Pre: y is defined. 
//Post: Converts y from an image array pixel index to the 
//      imaginary part of the complex graph location that the
//      pixel represents. 
//      NOTE: The y axis is inverted. (i.e. y = 0 is top of image)
inline __device__ float pixelYToComplexImag(uint y) {
  return((-(y/((float) WINDOW_HEIGHT)) * VERT_SIZE) + START_Y + (((float) VERT_SIZE) / 2));
}

//Pre: x and y are defined and are the matrix indices of an image.
//Post: Computes the pixel value for the Mandelbrot set at 
//      the given pixel.
inline __device__ uchar getPixelValue(uint x, uint y) {
  float real = pixelXToComplexReal(x);
  float imag = pixelYToComplexImag(y);
  float init_real = real;
  float init_imag = imag;
  int i;
  for(i = 0; i < MAX_DEPTH; i++) {
    if(ABS(real, imag) > EXCEED_VALUE)
      break;
    real = MAND_REAL(real, imag, init_real);
    imag = MAND_IMAG(real, imag, init_imag);
  }
  uchar value = (uchar) ((i / ((float)MAX_DEPTH)) * COLOR_MAX);
  return(value);
}

//Pre: image is defined and has length lenImage. 
//Post: Modifies the elements in image to be a grayscale Mandelbrot
//      image. 
__global__ void mand(uchar * image, int lenImage) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if(i < lenImage) {
    int x = i % WINDOW_WIDTH;
    int y = i / WINDOW_WIDTH;
    image[i] = getPixelValue(x, y);
  }
}
""")


#Pre: image is a 1D array of length WINDOW_HEIGHT * WINDOW_WIDTH
#Post: Returns None.
def printASCIISet(image):
    row = 0
    col = 0
    for i in range(WINDOW_HEIGHT * WINDOW_WIDTH):
        #print image[i]
        if int(image[i]) > 225:
            print "O",
        elif image[i] > 10:
            print "o",
        elif image[i] > 5:
            print ".",
        else:
            print " ",
        col += 1
        if col == WINDOW_WIDTH:
            print
            row += 1
            col = 0

NUM_THREADS = 512

if __name__ == "__main__":
    # Uncomment to get rid of timing the runtime initialization:
    # ===
    # int * dummy;
    # cudaMalloc(&dummy, 0);
    # cudaFree(dummy);
    # ===
    # boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
    t1 = time.time()
    lenImage = WINDOW_HEIGHT * WINDOW_WIDTH
    # Create a greyscale image on HOST:
    image = numpy.zeros(lenImage, dtype=numpy.uint8)
    # Create a greyscale image on GPU:
    gpuImage = cuda.mem_alloc(image.nbytes)
    numBlocks = (lenImage // NUM_THREADS) + 1
    mand = module.get_function("mand")
    mand(gpuImage, block=(NUM_THREADS,1,1), grid=(numBlocks, 1))
    cuda.memcpy_dtoh(image, gpuImage)
    
    # TEMPORARY CHECK:
    # ===
    printASCIISet(image)
    # ===
    
    # Cleanup...
    # delete[] image
    # cudaFree(gpuImage)
    # compute time:
    t2 = time.time()
    print t2 - t1
