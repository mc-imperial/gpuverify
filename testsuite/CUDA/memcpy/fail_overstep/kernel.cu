//xfail:NOT_ALL_VERIFIED
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is racy.
//
//It uses uses memcpy and copies too many bytes.

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

typedef struct {
  short x;
  short y;
  char z;
} s_t; //< sizeof(s_t) == 6

__global__ void k(s_t *in, s_t *out) {
  memcpy(&out[threadIdx.x], &in[threadIdx.x], 12); //< copy two elements
}
