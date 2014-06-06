#include <iostream>
#include "constants.h"
//#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;
using namespace PARAMS;

//Pre: x is defined. 
//Post: Converts x from an image array pixel index to 
//      the real part of the complex graph location that the
//      pixel represents. 
inline __device__ float pixelXToComplexReal(uint x, int w) {
  return(((x / ((float) w)) * SIZE) + START_X - (SIZE / 2.f));
}

//Pre: y is defined. 
//Post: Converts y from an image array pixel index to the 
//      imaginary part of the complex graph location that the
//      pixel represents. 
//      NOTE: The y axis is inverted. (i.e. y = 0 is top of image)
inline __device__ float pixelYToComplexImag(uint y, int h, float vert_size) {
  return((-(y/((float) h)) * vert_size) + START_Y + (vert_size / 2.f));
}

//Pre: x and y are defined and are the matrix indices of an image.
//     w, h are width and height of image.
//     max_depth is the maximum recursive formula depth. 
//     vert_size is the complex vertical size of the window.
//Post: Computes the pixel value for the Mandelbrot set at 
//      the given pixel.
inline __device__ uchar getPixelValue(uint x, uint y, int w, int h, 
				      int max_depth, float vert_size) {
  float real = pixelXToComplexReal(x, w);
  float imag = pixelYToComplexImag(y, h, vert_size);
  float init_real = real;
  float init_imag = imag;
  int i;
  for(i = 0; i < max_depth; i++) {
    if(ABS(real, imag) > EXCEED_VALUE)
      break;
    real = MAND_REAL(real, imag, init_real);
    imag = MAND_IMAG(real, imag, init_imag);
  }
  uchar value = (uchar) ((i / ((float)max_depth)) * COLOR_MAX);
  return(value);
}

//Pre: image is defined and has length lenImage. 
//     w, h are width and height of image.
//     max_depth is the maximum recursive formula depth. 
//     vert_size is the complex vertical size of the window.
//Post: Modifies the elements in image to be a grayscale Mandelbrot
//      image. 
__global__ void mand(uchar * image, int lenImage, int w, int h, 
		     int depth, float vert_size) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if(i < lenImage) {
    int x = i % w;
    int y = i / w;
    image[i] = getPixelValue(x, y, w, h, depth, vert_size);
  }
}

void printASCIISet(uchar * image) {
  int row = 0;
  int col = 0;
  for(int i = 0; i < height * width; i++) {
    if(image[i] > 225)
      cout << "O ";
    else if(image[i] > 50)
      cout << "o  ";
    else if(image[i] > 5)
      cout << ". ";
    else
      cout << "  ";
    col++;
    if(col == width) {
      cout << endl;
      row++;
      col = 0;
    }
    //cout << (int) image[row][col] << endl;
  }
}

//#define BLOCK_SIZE 16
#define NUM_THREADS 512

int main(int argc, char ** argv) {
  if(argc == 4) {
    setParams(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  }
  //Uncomment to get rid of timing the runtime initialization:
  //===
  // int * dummy;
  // cudaMalloc(&dummy, 0);
  // cudaFree(dummy);
  //===
  //  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  int lenImage = height * width;
  //Create a greyscale image on HOST:
  uchar * image = (uchar*) malloc(sizeof(uchar) * lenImage);
  //Create a greyscale image on GPU:
  uchar * gpuImage;
  cudaMalloc(&gpuImage, sizeof(uchar) * lenImage);
  //define the number of blocks:
  int numBlocks = (lenImage / NUM_THREADS) + 1;
  //Activate kernel
  mand<<<numBlocks, NUM_THREADS>>> (gpuImage, lenImage, width, height,
				    max_depth, vert_size);
  //Copy mand image back to host
  cudaMemcpy(image, gpuImage, sizeof(uchar) * lenImage, 
	     cudaMemcpyDeviceToHost);
  
  //TEMPORARY CHECK:
  //===
  //printASCIISet(image);
  //===
  
  //Cleanup...
  delete[] image;
  cudaFree(gpuImage);
  //Compute time:
  // boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  // boost::posix_time::time_duration duration = t2 - t1;
  // long micro = duration.total_microseconds();
  // double sec = micro / 1000000.;
  // cout << sec << endl;
  return(0);
}
