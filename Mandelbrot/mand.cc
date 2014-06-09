#include <iostream>
#include "constants.h"
#include <stdlib.h>
//#include "boost/date_time/posix_time/posix_time.hpp"

typedef unsigned char uchar;
typedef unsigned int uint;

using namespace std;
using namespace PARAMS;


//Pre: x is defined. 
//Post: Converts x from an image array pixel index to 
//      the real part of the complex graph location that the
//      pixel represents. 
inline float pixelXToComplexReal(uint x) {
  return(((x / ((float) width)) * SIZE) + START_X - (SIZE / 2.f));
}

//Pre: y is defined. 
//Post: Converts y from an image array pixel index to the 
//      imaginary part of the complex graph location that the
//      pixel represents. 
//      NOTE: The y axis is inverted. (i.e. y = 0 is top of image)
inline float pixelYToComplexImag(uint y) {
  return((-(y/((float) height)) * vert_size) + START_Y + (((float) vert_size) / 2));
}

//Pre: x and y are defined and are the matrix indices of an image.
//Post: Computes the pixel value for the Mandelbrot set at 
//      the given pixel.
inline uchar getPixelValue(uint x, uint y) {
  float real = pixelXToComplexReal(x);
  float imag = pixelYToComplexImag(y);
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

void printASCIISet(uchar * image) {
  int row = 0;
  int col = 0;
  for(int i = 0; i < height * width; i++) {
    if(image[i] > 225)
      cout << "O ";
    else if(image[i] > 10)
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
  }
}

int main(int argc, char ** argv) {
  if(argc == 4) {
    setParams(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  }
  //  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  //Create a greyscale image:
  uchar * image = (uchar*) malloc(sizeof(uchar) * height * width);
  int row = 0;
  int col = 0;
  for(int i = 0; i < width * height; i++) {
    image[i] = getPixelValue(col, row);
    col++;
    if(col == width) {
      row++;
      col = 0;
    }
  }
  //TEMPORARY CHECK:
  //===
  // printASCIISet(image);
  //===
  
  //Cleanup...
  delete[] image;
  //Compute time:
  // boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  // boost::posix_time::time_duration duration = t2 - t1;
  // long micro = duration.total_microseconds();
  // double sec = micro / 1000000.;
  // cout << sec << endl;
  return(0);
}
