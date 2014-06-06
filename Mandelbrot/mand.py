#import time
#t1 = time.time()
from constants import *
import sys

#Pre: x is defined. 
#Post: Converts x from an image array pixel index to 
#      the real part of the complex graph location that the
#      pixel represents. 
#      Returns a float.
def pixelXToComplexReal(x):
    return ((x / float(WINDOW_WIDTH) * SIZE)) + START_X - (SIZE / 2)

#Pre: y is defined. 
#Post: Converts y from an image array pixel index to the 
#      imaginary part of the complex graph location that the
#      pixel represents. 
#      NOTE: The y axis is inverted. (i.e. y = 0 is top of image)
#      Returns a float. 
def pixelYToComplexImag(y):
    return (-(y / float(WINDOW_HEIGHT) * VERT_SIZE)) + START_Y + ((float(VERT_SIZE) / 2))

#Pre: x and y are defined and are the matrix indices of an image.
#Post: Computes the pixel value for the Mandelbrot set at 
#      the given pixel.
def getPixelValue(x, y):
    real = pixelXToComplexReal(x);
    imag = pixelYToComplexImag(y);
    init_real = real;
    init_imag = imag;
    value = 0
    for i in range(MAX_DEPTH):
        if ABS(real, imag) > EXCEED_VALUE:
            value = i
            break
        real = MAND_REAL(real, imag, init_real)
        imag = MAND_IMAG(real, imag, init_imag)
    else:
        value = MAX_DEPTH
    value = (i / float(MAX_DEPTH)) * COLOR_MAX
    return value

#Pre: image is a 1D array of length WINDOW_HEIGHT * WINDOW_WIDTH
#Post: Returns None.
def printASCIISet(image):
    row = 0
    col = 0
    for i in range(WINDOW_HEIGHT * WINDOW_WIDTH):
        if image[i] > 225:
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

#if __name__ == "__main__":
if len(sys.argv) == 4:
    WINDOW_WIDTH = int(sys.argv[1])
    WINDOW_HEIGHT = int(sys.argv[2])
    MAX_DEPTH = int(sys.argv[3])
    VERT_SIZE = SIZE * (float(WINDOW_HEIGHT) / float(WINDOW_WIDTH))
#Create a greyscale image:
image = [0] * (WINDOW_HEIGHT * WINDOW_WIDTH)
row = 0
col = 0
for i in range(WINDOW_WIDTH * WINDOW_HEIGHT):
    image[i] = getPixelValue(col, row)
    col += 1
    if col == WINDOW_WIDTH:
        row += 1
        col = 0
#TEMPORARY CHECK:
#===
#printASCIISet(image)
#===

#Compute time:
# t2 = time.time()
# duration = t2 - t1
# print duration
