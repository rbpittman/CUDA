#define DEFAULT_N 16

#define RANDOM_SEED 37
#define A_VALUE 19609
#define B_VALUE 171
#define M_VALUE 3301

#define NEXT_RANDOM(seed) (((seed * A_VALUE) + B_VALUE) % M_VALUE)
