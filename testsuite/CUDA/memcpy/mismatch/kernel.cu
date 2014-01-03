//pass
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is race-free.
//
//The memcpy is between different src and dst types so we have to handle the
//arrays in and out at the byte-level.

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

typedef struct {
  short x;
  char y;
} s1_t; //< sizeof(s1_t) == 4

typedef struct {
  short x;
  short y;
} s2_t; //< sizeof(s2_t) == 4

__global__ void k(s1_t *in, s2_t *out) {
  size_t len = 4;
  memcpy(&out[threadIdx.x], &in[threadIdx.x], len);
}
