#include <iostream>
#include "constants.h"
#include "boost/date_time/posix_time/posix_time.hpp"

#define NUM_BLOCKS 2
#define START_NUM_THREADS 0
#define END_NUM_THREADS 1024
#define STEP 16
#define DEPTH 223456789
#define TRIALS 2

#define TEST_ONE_THREAD true

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
    float oldReal = real;
    real = MAND_REAL(real,    imag, init_real);
    imag = MAND_IMAG(oldReal, imag, init_imag);
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
  //Use normal write location,
  int globalMemIdx = threadIdx.x + (blockDim.x * blockIdx.x);
  //but all threads compute the same Mandelbrot location:
  int i = 0;
  if(i < lenImage) {
    int x = i % w;
    int y = i / w;
    image[globalMemIdx] = getPixelValue(x, y, w, h, depth, vert_size);
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
// #define NUM_THREADS 512

#include <fstream>

void writeToFile(uchar * image, char * filename) {
  ofstream out(filename);
  out << "P6\n";
  out << width << ' ' << height << endl;
  out << "255" << endl;
  // unsigned char curr;
  // unsigned int count = 0;
  for(int i = 0; i < width * height; i++) {
    out << image[i] << image[i] << image[i];
    // if(count == 0) {
    //   curr = image[i];
    //   count = 1;
    // } else if(curr != image[i]) {
    //   out << count << endl;
    //   out << (int) curr << endl;
    //   count = 0;
    // } else {
    //   ++count;
    // }
  }
  // out << 0 << endl;
}

void printMandValue(uchar * gpuImage) {
  uchar * image = new uchar(123);
  cudaMemcpy(image, gpuImage, sizeof(uchar), cudaMemcpyDeviceToHost);
  cout << "Mandelbrot pixel: " << (int) image[0] << endl;
  delete(image);
}

//numThreads is the total number of threads, including those in separate
//blocks.
double runTest(int numThreads) {
  int numBlocks = NUM_BLOCKS;
  if(TEST_ONE_THREAD) {
    numThreads = 1;
    numBlocks = 1;
  } else {
    numThreads /= numBlocks;
  }
  //Create a greyscale image on GPU:
  uchar * gpuImage;
  cudaMalloc(&gpuImage, sizeof(uchar) * numThreads);
  //define the number of blocks:
  // int numBlocks = (lenImage / NUM_THREADS) + 1;
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  //Activate kernel
  // cout << "Blocks: " << NUM_BLOCKS << endl;
  // cout << "Threads: " << numThreads << endl;
  mand<<<numBlocks, numThreads>>> (gpuImage, numThreads, width, height,
				    max_depth, vert_size);
  cudaDeviceSynchronize();
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  //=============
  // printMandValue(gpuImage);
  //=============
  cudaFree(gpuImage);
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  return (micro / 1000000.);
}

int main(int argc, char ** argv) {
  setParams(1, 1, DEPTH);
  //Uncomment to get rid of timing the runtime initialization:
  //===
  int * dummy;
  cudaMalloc(&dummy, 0);
  cudaFree(dummy);
  //===
  if(TEST_ONE_THREAD) {
    cout << "Testing one thread" << endl;
    cout << runTest(1) << endl;
  } else {
    cout << "Bench for Mandelbrot parallelality test. x is number of threads executing\n";
    cout << "Trials: " << TRIALS << endl;
    for(int numThreads = START_NUM_THREADS; numThreads < END_NUM_THREADS + 1;
	numThreads += STEP) {
      cout << numThreads << endl;
      for(int i = 0; i < TRIALS; i++) {
	cout << "0m" << runTest(numThreads);
	if(i != TRIALS - 1) {
	  cout << ' ';
	}
      }
      cout << endl;
    }
  }
  return(0);
}
