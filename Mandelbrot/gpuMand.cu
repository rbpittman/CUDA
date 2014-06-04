#include <iostream>
#include "constants.h"
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

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

void printASCIISet(uchar * image) {
  int row = 0;
  int col = 0;
  for(int i = 0; i < WINDOW_HEIGHT * WINDOW_WIDTH; i++) {
    if(image[i] > 225)
      cout << "O ";
    else if(image[i] > 50)
      cout << "o  ";
    else if(image[i] > 5)
      cout << ". ";
    else
      cout << "  ";
    col++;
    if(col == WINDOW_WIDTH) {
      cout << endl;
      row++;
      col = 0;
    }
    //cout << (int) image[row][col] << endl;
  }
}

//#define BLOCK_SIZE 16
#define NUM_THREADS 512

int main() {
  //Uncomment to get rid of timing the runtime initialization:
  //===
  int * dummy;
  cudaMalloc(&dummy, 0);
  cudaFree(dummy);
  //===
  
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  int lenImage = WINDOW_HEIGHT * WINDOW_WIDTH;
  //Create a greyscale image on HOST:
  uchar * image = (uchar*) malloc(sizeof(uchar) * lenImage);
  //Create a greyscale image on GPU:
  uchar * gpuImage;
  //cerr << "START" << endl;
  cudaMalloc(&gpuImage, sizeof(uchar) * lenImage);
  //cerr << "END" << endl;
  /*SQUARE WAY OF DOING THE THREADS
  //Block has 16X16 threads:
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  //Our grid of blocks has to at least cover the entire image. 
  //A check will be done in the kernel to see if the pixel is in range.
  int gridWidth = WINDOW_WIDTH / BLOCK_SIZE;
  if(WINDOW_WIDTH % BLOCK_SIZE != 0) gridWidth++;
  int gridHeight = WINDOW_HEIGHT / BLOCK_SIZE;
  if(WINDOW_HEIGHT % BLOCK_SIZE != 0) gridHeight++;
  dim3 dimGrid(gridWidth, gridHeight);
  mand<<<dimGrid, dimBlock>>> (gpuImage);
  */
  int numBlocks = (lenImage / NUM_THREADS) + 1;
  mand<<<numBlocks, NUM_THREADS>>> (gpuImage, lenImage);
  
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
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  double sec = micro / 1000000.;
  cout << sec << endl;
  return(0);
}
