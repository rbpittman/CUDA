//Particles can go off screen, their position will still be
//defined, but not drawn.

#include <iostream>
#include <fstream>
#include <math.h> //For roundf
#include "constants.h"

#include <boost/timer.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;

//Pre: in is defined and contains num_particles lines. 
//Post: Returns the coordinates from the file as a single
//      array. even is x, odd is y. 
//      length of array == 2 * num_particles.
// float * readParticles(ifstream & in, const int & num_particles) {
//   float * list = new float[2 * num_particles];
//   for(int i = 0; i < num_particles; i++) {
//     int x;
//     in >> x;
//     int y;
//     in >> y;
//     list[2 * i] = x;
//     list[(2 * i) + 1] = y;
//   }
//   return(list);
// }

//The magnitude within which both the particle x and y must
//be between in order to be simulated. 
#define BOUNDS 100000
#define GRAV_FACTOR 1.f

//force formula: f = 1/(d^2)
//Note: f = 1/d would be much more computionally efficient.
//Given force vector f (assuming mass is 1 for F = M*A), add f
//components to particle x and y. 
void runframe(const float * positions, float * resultPositions, float * vels,
	      bool * inBounds, const int & numParts) {
  for(int threadIdx = 0; threadIdx < numParts; threadIdx++) {
    float x_pos = positions[2 * threadIdx];
    float y_pos = positions[(2 * threadIdx) + 1];
    inBounds[threadIdx] = ((-BOUNDS <= x_pos) && (x_pos <= BOUNDS) &&
			   ((-BOUNDS <= y_pos) && (y_pos <= BOUNDS)));
    if(inBounds[threadIdx]) {
      //Iterate over other particles:
      for(int i = 0; i < numParts; i++) {
	//Make sure it's not the same particle and it's in-bounds:
	if((i != threadIdx) && (inBounds[i])) {
	  //Get other particle position:
	  float other_x = positions[2 * i];
	  float other_y = positions[2 * i + 1];
	  //Get force x:
	  float delta_x = other_x - x_pos;
	  float delta_y = other_y - y_pos;
	  float force_denom = (delta_x * delta_x) + (delta_y * delta_y);
	  //uncomment:f = 1/(d^2), comment: f = 1/d
	  //float denom = /*sqrt(force_denom) * */force_denom;
	  if(force_denom != 0) {
	    float x_force = (delta_x * GRAV_FACTOR) / force_denom;
	    float y_force = (delta_y * GRAV_FACTOR) / force_denom;
	    vels[2 * threadIdx] += x_force;
	    vels[(2 * threadIdx) + 1] += y_force;
	  }
	}
      }
      //    }
    //Sync:
    //  syncthreads();
      //    if(inBounds[threadIdx]) {
      //Update all positions:
      resultPositions[2 * threadIdx] = positions[2 * threadIdx] + vels[2 * threadIdx];
      resultPositions[(2 * threadIdx) + 1] = positions[(2 * threadIdx) + 1] + vels[(2 * threadIdx) + 1];
    }
  }
}

//We're going to say that an n size array of positions (x,y) or
//velocities is actually a 2 * n size array, where each even index
//is the x coord, and odd is y. 

//Pre: out is defined and available for writing. 
//     positions has length num * 2. 
//Post: writes the int rounded position of each particle to
//      the file, with a particle position per line. 
void write_frame(ofstream & out, float * positions, int num) {
  for(int i = 0; i < num; i++) {
    out << roundf(positions[2 * i]) << ' '
	<< roundf(positions[(2 * i) + 1]) << endl;
  }
}

//Pre: out is defined. 
//Post: Writes to the file:
//      WINDOW_WIDTH WINDOW_HEIGHT
//      num_particles
void write_header(ofstream & out, int num_particles) {
  out << WINDOW_WIDTH << ' ' << WINDOW_HEIGHT << endl;
  out << num_particles << endl;
}

inline float * copyArray(float * array, int size) {
  float * newArray = new float[size];
  for(int i = 0; i < size; i++) {
    newArray[i] = array[i];
  }
  return(newArray);
}

#define RANDOM_SEED 37
#define A 19609
#define B 171
#define M 3301

#define NEXT_RANDOM(seed) (((seed * A) + B) % M)// % RAND_RANGE)

#define MAX_DIST 500
#define SHIFT(pos) (pos % MAX_DIST)

//Post: modifies seed to be the next seed. 
//      Returns a pseudo-random number. 
inline int nextRandom(int & seed) {
  seed = NEXT_RANDOM(seed);
  return(seed);
}

inline float * getParticles(int num) {
  int seed = RANDOM_SEED;
  float * parts = new float[2 * num];
  for(int i = 0; i < 2 * num; i++) {
    parts[i] = SHIFT(nextRandom(seed));
  }
  return(parts);
}

#define ARG_IS_NUM_PARTICLES true

//Pre: PARTICLE_FILE contains less than 512 particles. 
int main(int argc, char ** argv) {
  //ifstream partFile(PARTICLE_FILE);
  if(argc != 2) {
    cout << "ERROR: expected number of particles as argument.\n";
  } else {
    int num_particles = NUM_PARTICLES;
    int num_frames = NUM_FRAMES;
    if(ARG_IS_NUM_PARTICLES) 
      num_particles = atoi(argv[1]);
    else
      num_frames = atoi(argv[1]);
    //partFile >> num_particles;
    
    //Get host memory positions:
    float * positions = getParticles(num_particles);
    //readParticles(partFile,num_particles)
    
    //Allocate and initialize host velocities:
    float * vels = new float[2 * num_particles];
    for(int i = 0; i < 2 * num_particles; i++) vels[i] = 0;
    bool * inBounds = new bool[num_particles];
    //for(int i = 0; i < num_particles; i++) inBounds[i] = true;
    memset(inBounds, false, num_particles * sizeof(bool));
    float ** frames = new float*[num_frames];
    //float * currFrame = new float[2 * num_particles];
    for(int i = 0; i < num_frames; i++) {
      frames[i] = new float[2 * num_particles];
      if(i == 0) {
	runframe(positions, frames[i], vels, inBounds, num_particles);
      } else {
	runframe(frames[i - 1], frames[i], vels, inBounds, num_particles);
      }
    }
    
    //Store results:
    // ofstream out(OUTPUT_FILE);
    // write_header(out, num_particles);
    // for(int i = 0; i < num_frames; i++) {
    //   write_frame(out, frames[i], num_particles);
    // }
    // out.close();
    
    //Clean up...
    //partFile.close();
    delete[] inBounds;
    delete[] vels;
    delete[] positions;
  }
  return(0);
}
