#include <iostream>
#include "constants.h"
#include "boost/date_time/posix_time/posix_time.hpp"

#define DEPTH 223456789

using namespace std;
using namespace PARAMS;

//Pre: x is defined. 
//Post: Converts x from an image array pixel index to 
//      the real part of the complex graph location that the
//      pixel represents. 
inline float pixelXToComplexReal(uint x, int w) {
  return(((x / ((float) w)) * SIZE) + START_X - (SIZE / 2.f));
}

//Pre: y is defined. 
//Post: Converts y from an image array pixel index to the 
//      imaginary part of the complex graph location that the
//      pixel represents. 
//      NOTE: The y axis is inverted. (i.e. y = 0 is top of image)
inline float pixelYToComplexImag(uint y, int h, float vert_size) {
  return((-(y/((float) h)) * vert_size) + START_Y + (vert_size / 2.f));
}

//Pre: x and y are defined and are the matrix indices of an image.
//     w, h are width and height of image.
//     max_depth is the maximum recursive formula depth. 
//     vert_size is the complex vertical size of the window.
//Post: Computes the pixel value for the Mandelbrot set at 
//      the given pixel.
inline uchar getPixelValue(uint x, uint y, int w, int h, 
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
inline void mand(uchar * image, int lenImage, int w, int h, int depth, float vert_size) {
  int i = 0;
  if(i < lenImage) {
    int x = i % w;
    int y = i / w;
    image[i] = getPixelValue(x, y, w, h, depth, vert_size);
  }
}

double runTest() {
  uchar * image = new uchar[1];
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  mand(image, 1, width, height, max_depth, vert_size);
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  cout << "value: " << (int) image[0] << endl;
  boost::posix_time::time_duration duration = t2 - t1;
  delete[] image;
  long micro = duration.total_microseconds();
  return (micro / 1000000.);
}

int main(int argc, char ** argv) {
  setParams(1, 1, DEPTH);
  cout << runTest() << endl;
  return(0);
}
