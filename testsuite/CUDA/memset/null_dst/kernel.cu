//xfail:NOT_ALL_VERIFIED
//--gridDim=1 --blockDim=32 --no-inline

#define memset(dst,val,len) __builtin_memset(dst,val,len)

__device__ int bar(void);

__global__ void kernel(uint4 *out) {
  memset(0, 0, 16);
}
