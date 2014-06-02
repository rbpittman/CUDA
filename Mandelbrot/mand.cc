#include <iostream>
#include "constants.h"
#include "boost/date_time/posix_time/posix_time.hpp"

typedef unsigned char uchar;
typedef unsigned int uint;

using namespace std;

//Pre: x is defined. 
//Post: Converts x from an image array pixel index to 
//      the real part of the complex graph location that the
//      pixel represents. 
inline float pixelXToComplexReal(uint x) {
  return(((x / ((float) WINDOW_WIDTH)) * SIZE) + START_X - (SIZE / 2.f));
}

//Pre: y is defined. 
//Post: Converts y from an image array pixel index to the 
//      imaginary part of the complex graph location that the
//      pixel represents. 
//      NOTE: The y axis is inverted. (i.e. y = 0 is top of image)
inline float pixelYToComplexImag(uint y) {
  return((-(y / ((float) WINDOW_HEIGHT)) * VERT_SIZE) + START_Y + (((float) VERT_SIZE) / 2));
}

//Pre: x and y are defined and are the matrix indices of an image.
//Post: Computes the pixel value for the Mandelbrot set at 
//      the given pixel.
uchar getPixelValue(uint x, uint y) {
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
  uchar value = (uchar) ((i / ((float)MAX_DEPTH)) * UCHAR_MAX);
  return(value);
}

int main() {
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  //Create a greyscale image:
  uchar ** image = new uchar*[WINDOW_HEIGHT];
  for(int row = 0; row < WINDOW_HEIGHT; row++) {
    image[row] = new uchar[WINDOW_WIDTH];
    //Compute Mandelbrot set values:
    for(int col = 0; col < WINDOW_WIDTH; col++) {
      image[row][col] = getPixelValue(col, row);
    }
  }
  
  //Cleanup...
  for(int i = 0; i < WINDOW_HEIGHT; i++) {
    delete[] image[i];
  }
  delete[] image;
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration duration = t2 - t1;
  long micro = duration.total_microseconds();
  double sec = micro / 1000000.;
  cout << sec << endl;
  return(0);
}
