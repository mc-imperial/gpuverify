//--gridDim=[2,2] --blockDim=[4,2]

// OOPSLA 2014 Transpose example
// NB: we have added preconditions to the figure.

#define TILE_DIM 4
#define BLOCK_ROWS 2

__global__ void transpose(
  float *odata, float *idata, int width, int height) {
  __requires(width == 8);
  __requires(height == 8);

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    odata[index_out+i] = idata[index_in+i*width];
}
