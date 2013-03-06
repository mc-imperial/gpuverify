//pass
//--blockDim=64 --gridDim=64

#include <cuda.h>

typedef struct {
  float x,y,z,w;
} myfloat4;

__global__ void k() {
  myfloat4 f4;
  float i0 = f4.x;
}
