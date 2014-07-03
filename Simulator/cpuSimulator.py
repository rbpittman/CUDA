import sys
from constants import *

BOUNDS = 100000
GRAV_FACTOR = 1.

#force formula: f = 1/(d^2)
#Note: f = 1/d would be much more computionally efficient.
#Given force vector f (assuming mass is 1 for F = M*A), add f
#components to particle x and y. 
def runframe(positions, resultPositions, vels, inBounds, numParts):
    for threadIdx in range(numParts):
        x_pos = positions[2 * threadIdx]
        y_pos = positions[(2 * threadIdx) + 1]
        inBounds[threadIdx] = (-BOUNDS <= x_pos <= BOUNDS) and (-BOUNDS <= y_pos <= BOUNDS)
        if inBounds[threadIdx]:
            #Iterate over other particles:
            for i in range(numParts):
                if (i != threadIdx) and inBounds[i]:
                    other_x = positions[2 * i]
                    other_y = positions[2 * i + 1]
                    delta_x = other_x - x_pos
                    delta_y = other_y - y_pos
                    force_denom = (delta_x * delta_x) + (delta_y * delta_y)
                    if force_denom != 0:
                        x_force = (delta_x * GRAV_FACTOR) / force_denom
                        y_force = (delta_y * GRAV_FACTOR) / force_denom
                        vels[2 * threadIdx] += x_force
                        vels[(2 * threadIdx) + 1] += y_force
    #Sync:
#        if inBounds[threadIdx]:
        resultPositions.append(positions[2 * threadIdx] + vels[2 * threadIdx])
        resultPositions.append(positions[(2 * threadIdx) + 1] + vels[(2 * threadIdx) + 1])
#We're going to say that an n size array of positions (x,y) or
#velocities is actually a 2 * n size array, where each even index
#is the x coord, and odd is y. 

#Pre: out is defined and available for writing. 
#     positions has length num * 2. 
#Post: writes the int rounded position of each particle to
#      the file, with a particle position per line. 
def write_frame(out, positions, num):
    for i in range(num):
        out.write(str(int(round(positions[2 * i]))) + ' ')
        out.write(str(int(round(positions[(2 * i) + 1]))) + '\n')
        
#Pre: out is defined. 
#Post: Writes to the file:
#      WINDOW_WIDTH WINDOW_HEIGHT
#      num_particles
def write_header(out, num_particles):
    out.write(str(WINDOW_WIDTH) + ' ' + str(WINDOW_HEIGHT) + '\n')
    out.write(str(num_particles) + '\n')


RANDOM_SEED = 37
A = 19609
B = 171
M = 3301

NEXT_RANDOM = lambda seed: (((seed * A) + B) % M)# % RAND_RANGE)

MAX_DIST = 500
SHIFT = lambda pos: (pos % MAX_DIST)

#Post: modifies seed to be the next seed. 
#      Returns a pseudo-random number. 
def nextRandom(seed):
    return NEXT_RANDOM(seed)

def getParticles(num):
    seed = RANDOM_SEED
    #parts = new float[2 * num]
    parts = []
    for i in range(2 * num):
        seed = nextRandom(seed)
        parts.append(SHIFT(seed))
    return parts


ARG_IS_NUM_PARTICLES = True

#Pre: PARTICLE_FILE contains less than 512 particles. 
if __name__ == "__main__":
    # ifstream partFile(PARTICLE_FILE)
    if len(sys.argv) != 2:
        print "ERROR: expected number of particles as argument.\n"
    else:
        num_particles = NUM_PARTICLES
        num_frames = NUM_FRAMES
        if ARG_IS_NUM_PARTICLES:
            num_particles = int(sys.argv[1])
        else:
            num_frames    = int(sys.argv[1])
        positions = getParticles(num_particles)
        vels = [0.] * (2 * num_particles)
        inBounds = [True] * num_particles
        frames = [[] for i in range(num_frames)]
        # float * currFrame = new float[2 * num_particles]
        for i in range(num_frames):
            # frames[i] = [0] * (num_particles * 2)
            if i == 0:
                runframe(positions, frames[i], vels, inBounds, num_particles)
            else:
                runframe(frames[i - 1], frames[i], vels, inBounds, num_particles)
        # Store results:
        # out = open("pySim.txt", 'w')
        # write_header(out, num_particles)
        # for i in range(num_frames):
        #     write_fprame(out, frames[i], num_particles)
        # out.close()
