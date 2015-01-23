//xfail:NOT_ALL_VERIFIED
//--gridDim=1 --blockDim=2 --no-inline

//This kernel has a null pointer access.

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

typedef struct {
  short x;
  short y;
} s_t; //< sizeof(s2_t) == 4

__global__ void k(s_t *in, s_t *out) {
  memcpy(0, &in[threadIdx.x], sizeof(s_t));
}
