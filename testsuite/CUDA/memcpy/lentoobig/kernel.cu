//xfail:NOT_ALL_VERIFIED
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is racy.
//
//The memcpy resolves to a non-integer number of element writes so we have to
//handle the arrays in and out at the byte-level.

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

typedef struct {
  short x;
  short y;
} s_t; //< sizeof(s_t) == 4

__global__ void k(s_t *in, s_t *out) {
  size_t len = 5;
  memcpy(&out[threadIdx.x], &in[threadIdx.x], len);
}
