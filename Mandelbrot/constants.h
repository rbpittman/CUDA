//The max number of iterations before stopping the recursive
//Mandelbrot function:
#define MAX_DEPTH 500
#define WINDOW_WIDTH 50
#define WINDOW_HEIGHT 50
//DO NOT TOUCH ANY OF THE ABOVE LINES, constants.py READS THOSE VALUES. 
//Horizontal size in the complex plane:
#define SIZE ((float) 4)
#define VERT_SIZE SIZE * (((float)WINDOW_HEIGHT) / ((float) WINDOW_WIDTH))
#define START_X ((float)0)
#define START_Y ((float)0)

#define COLOR_MAX 255

//Stop computing Mandelbrot function if this value is exceeded:
#define EXCEED_VALUE 2

#define ABS(x,y) (x * x) + (y * y)

//Pre: (zr, zi) are (real, imag) of current mandelbrot iteration.
//     r is the initial real value.
//Post: Returns the real part of the next iteration of the formula. 
#define MAND_REAL(zr, zi, r) (zr * zr) - (zi * zi) + r
//Same, except the imaginary part. 
#define MAND_IMAG(zr, zi, i) (2 * zr * zi) + i

typedef unsigned char uchar;
typedef unsigned int uint;

