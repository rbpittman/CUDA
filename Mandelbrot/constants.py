f = open("constants.h", 'r')
f.readline()
f.readline()
MAX_DEPTH = int(f.readline().split(" ")[-1])
WINDOW_WIDTH = int(f.readline().split(" ")[-1])
WINDOW_HEIGHT = int(f.readline().split(" ")[-1])
f.close()

#Horizontal size in the complex plane:
SIZE = float(4)
VERT_SIZE = SIZE * (float(WINDOW_HEIGHT) / float(WINDOW_WIDTH))
START_X = 0
START_Y = 0

COLOR_MAX = 255


#Stop computing Mandelbrot function if this value is exceeded:
EXCEED_VALUE = 2

ABS = lambda x,y: (x * x) + (y * y)

#Pre: (zr, zi) are (real, imag) of current mandelbrot iteration.
#     r is the initial real value.
#Post: Returns the real part of the next iteration of the formula. 
MAND_REAL = lambda zr, zi, r: (zr * zr) - (zi * zi) + r
#Same, except the imaginary part. 
MAND_IMAG = lambda zr, zi, i: (2 * zr * zi) + i
