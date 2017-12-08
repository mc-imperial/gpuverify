//pass
//--blockDim=2048 --gridDim=64

class s {
 int *p;

public:
 __device__ void store(int val) {
   p[threadIdx.x + blockDim.x * blockIdx.x] = val;
 }
};

__global__ void foo(s q) {
  q.store(42);
}
