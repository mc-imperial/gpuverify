//pass
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is race-free.
//
//It uses uses memcpy and copies fewer bytes than the struct size so we have to
//handle the arrays in and out at the byte-level.

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

typedef struct {
  short x;
  short y;
  char z;
} s_t; //< sizeof(s_t) == 6

__global__ void k(s_t *in, s_t *out) {
  size_t len = 4;
  memcpy(&out[threadIdx.x], &in[threadIdx.x], len);
}
