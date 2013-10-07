//xfail:BUGLE_ERROR
//--gridDim=1 --blockDim=32

//This kernel is not-racy: memset is called with variable value.
#define memset(dst,val,len) __builtin_memset(dst,val,len)

__global__ void kernel(uint4 *out) {
  uint4 vector; int val;
  memset(&vector, val, 16);
  out[threadIdx.x] = vector;
}
