import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from constants import *
import numpy, sys

#Post: Returns float32 numpy array. 
def readParticles(partFile, num_particles):
    l = []
    for i in range(num_particles):
        line = partFile.readline().split(" ")
        x = int(line[0])
        y = int(line[1])
        l.append(x)
        l.append(y)
    return numpy.array(l, dtype=numpy.float32)

#The magnitude within which both the particle x and y must
#be between in order to be simulated. 
BOUNDS = numpy.int32(100000)
GRAV_FACTOR = numpy.float32(1.0)

#Pre: positions is defined, and has length of at least (2*idx)+2.
#     The same goes for vels, but inBounds is at least idx + 1.
#Post: Computes the force on the idx'th particle in positions
#      (x = positions[2 * idx], y = positions[2 * idx + 1])
module = SourceModule("""
#define BOUNDS 100000
#define GRAV_FACTOR 1.f

//Pre: positions is defined, and has length of at least (2*idx)+2.
//     The same goes for vels, but inBounds is at least idx + 1.
//Post: Computes the force on the idx'th particle in positions
//      (x = positions[2 * idx], y = positions[2 * idx + 1])
__device__ inline void computeForce(float * positions, float * vels,
				    bool * inBounds, int idx,
				    int numParts) {
  float x_pos = positions[2 * idx];
  float y_pos = positions[2 * idx + 1];
  //Iterate over other particles:
  for(int i = 0; i < numParts; i++) {
    //Make sure it's not the same particle and it's in-bounds:
    if((i != idx) && (inBounds[i])) {
      //Get other particle position:
      float other_x = positions[2 * i];
      float other_y = positions[2 * i + 1];
      //Get force x:
      float delta_x = other_x - x_pos;
      float delta_y = other_y - y_pos;
      float denom = (delta_x * delta_x)+(delta_y * delta_y);
      //uncomment:f = 1/(d^2), comment: f = 1/d
      //      float denom = /*sqrt(force_denom) * */force_denom;
      if(denom != 0) {
	float x_force = (delta_x * GRAV_FACTOR) / denom;
	float y_force = (delta_y * GRAV_FACTOR) / denom;
	vels[2 * idx] += x_force;
	vels[2 * idx + 1] += y_force;
      }
    }
  }  
}


//force formula: f = 1/(d^2)
//Note: f = 1/d would be much more computionally efficient.
//Given force vector f (assuming mass is 1 for F = M*A), add f
//components to particle x and y. 
__global__ void runframe(float * positions, float * vels,
			 bool * inBounds, int numParts) {
  //Update all velocities:
  int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if(idx < numParts) {
    int x_idx = 2 * idx;
    int y_idx = x_idx + 1;
    float x_pos = positions[x_idx];
    float y_pos = positions[y_idx];
    inBounds[idx] = ((-BOUNDS <= x_pos) && (x_pos <= BOUNDS) &&
		     ((-BOUNDS <= y_pos) && (y_pos <= BOUNDS)));
    if(inBounds[idx]) {
      computeForce(positions, vels, inBounds, idx, numParts);
      //Update all positions:
      positions[x_idx] += vels[x_idx];
      positions[y_idx] += vels[y_idx];
    }
  }
}

""")

#We're going to say that an n size array of positions (x,y) or
#velocities is actually a 2 * n size array, where each even index
#is the x coord, and odd is y. 

#Pre: out is defined and available for writing. 
#     positions has length num * 2, and is a numpy array of int32. 
#Post: writes the int rounded position of each particle to
#      the file, with a particle position per line. 
def write_frame(out, positions, num):
    for i in range(num):
        out.write(str(int(round(positions[2 * i]))) + ' ' +
                  str(int(round(positions[(2 * i) + 1]))) + '\n')


#Pre: out is defined. 
#Post: Writes to the file:
#      WINDOW_WIDTH WINDOW_HEIGHT
#      num_particles
def write_header(out, num_particles):
    out.write(str(WINDOW_WIDTH) + ' ' + str(WINDOW_HEIGHT) + '\n')
    out.write(str(num_particles) + '\n')

def copy(array):
    pyList = []
    for i in range(len(array)):
        pyList.append(array[i])
    return pyList

NUM_THREADS = 23

#Pre: positions is a list of ints where len(positions) == 2*num_particles. 
#     num_particles <= 512
#Post: Runs a particle simulation for NUM_FRAMES steps with the given intial
#      particle positions. Each step is computed on the gpu. 
def execute(positions, num_particles, num_frames):
    #Get host positions:
    cpuPos = numpy.array(positions, dtype=numpy.float32)
    #Allocate position space on device:
    devPos = cuda.mem_alloc(cpuPos.nbytes)
    #Copy positions:
    cuda.memcpy_htod(devPos, cpuPos)
    
    #Allocate device velocities:
    devVels = cuda.mem_alloc(2 * num_particles * numpy.float32().nbytes)
    cuda.memset_d32(devVels, 0, 2 * num_particles)
    # #Copy velocities:
    # cuda.memcpy_htod(devVels, cpuVels)
    
    #Allocate and initialize device in bounds to false:
    #inBounds = numpy.zeros(num_particles, dtype=bool)
    devInBounds = cuda.mem_alloc(num_particles * numpy.bool8().nbytes)
    cuda.memset_d8(devInBounds, True, num_particles)
    
    # inB = numpy.zeros(num_particles, dtype=numpy.bool)
    # cuda.memcpy_dtoh(inB, devInBounds)
    # print inB
    
    # cuda.memcpy_htod(devInBounds, inBounds)
    # numBlocks = 1#(num_particles // 512) + 1;
    grid_dim = ((num_particles // NUM_THREADS) + 1, 1)
    print grid_dim
    runframe = module.get_function("runframe")
    frames = [None] * num_frames
    for i in range(num_frames):
        runframe(devPos, devVels, devInBounds, 
                 numpy.int32(num_particles),
                 grid=grid_dim,
                 block=(NUM_THREADS, 1, 1))
        #Get the positions from device:
        cuda.memcpy_dtoh(cpuPos, devPos)
        frames[i] = cpuPos.copy()
        #frames[i] = copy(cpuPos)
        #write_frame(out, cpuPos, num_particles)
    
    #Simulation destination file:
    # out = open(OUTPUT_FILE, 'w')
    # write_header(out, num_particles)
    # for frame in frames:
    #     write_frame(out, frame, num_particles)
    
    #clean up...
    #out.close()
    devPos.free()
    devVels.free()
    devInBounds.free()

MAX_DIST = 500
def getParticles(num):
    return numpy.random.randint(0, MAX_DIST, 2 * num)

ARG_IS_NUM_PARTICLES = True

#Pre: PARTICLE_FILE contains less than or equal to 512 particles. 
if __name__ == "__main__":
    #  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
    # partFile = open(PARTICLE_FILE, 'r')
    if len(sys.argv) != 2:
        print "Error: Expected arg"
    else:
        num_particles = NUM_PARTICLES
        num_frames = NUM_FRAMES
        if ARG_IS_NUM_PARTICLES:
            num_particles = int(sys.argv[1])
        else:
            num_frames    = int(sys.argv[1])
        #Get host memory positions:
        positions = getParticles(num_particles)
        execute(positions, num_particles, num_frames)
