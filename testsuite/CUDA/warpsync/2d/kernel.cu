//pass
//--blockDim=[4,4] --gridDim=[10,10] --warp-sync=16 --no-inline

#include <cuda.h>

#define SIZE 4
#define TILES 10
#define LENGTH (TILES * SIZE)

__global__ void matrix_transpose(float* A)
{
  __shared__ float tile [SIZE][SIZE];
  
  int x = threadIdx.x;
  int y = threadIdx.y;

  int tile_x = blockIdx.x;
  int tile_y = blockIdx.y;

	//tile[x][y] = A[((x + (tile_x * SIZE)) * LENGTH) + (y + (tile_y * SIZE))];

	tile[x][y] = tile[y][x];

	__syncthreads();

	A[((x + (tile_y * SIZE)) * LENGTH) + (y + (tile_x * SIZE))] = tile[x][y];
}
