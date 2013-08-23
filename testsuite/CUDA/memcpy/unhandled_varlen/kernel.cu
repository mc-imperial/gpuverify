//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=2

//This kernel is racy.
//
//However since the memcpy has a non-integer constant len we treat it as a
//no-op. This means we erroneously pass this test.

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

typedef struct {
  short x;
  short y;
  char z;
} s_t; //< sizeof(s_t) == 6

__global__ void overstep(s_t *in, s_t *out, size_t len) {
  memcpy(&out[threadIdx.x], &in[threadIdx.x], len);
}
