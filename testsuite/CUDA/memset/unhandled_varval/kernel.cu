//xfail:BUGLE_ERROR
//--gridDim=1 --blockDim=32 --no-inline

//This kernel is not-racy: memset is called with variable value.
#define memset(dst,val,len) __builtin_memset(dst,val,len)

__device__ int bar(void);

__global__ void kernel(uint4 *out) {
  uint4 vector;
  int val = bar();
  memset(&vector, val, 16);
  out[threadIdx.x] = vector;
}
