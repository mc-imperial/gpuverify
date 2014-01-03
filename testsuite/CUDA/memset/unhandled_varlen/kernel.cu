//xfail:BUGLE_ERROR
//--gridDim=1 --blockDim=32 --no-inline

//This kernel is racy: memset is called with variable length.
#define memset(dst,val,len) __builtin_memset(dst,val,len)

__device__ int bar(void);

__global__ void kernel(uint4 *out) {
  uint4 vector;
  int len = bar();
  memset(&vector, 0, len);
  out[threadIdx.x] = vector;
}
