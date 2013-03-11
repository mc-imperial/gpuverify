//pass
//--blockDim=[8,8] --gridDim=[1,1]

#include <cuda.h>


#define _2D_ACCESS(A, y, x, X_DIM) A[(y)*(X_DIM)+(x)]

#define X_DIMENSION 0
#define Y_DIMENSION 1

#define BLOCK_DIM (1 << 3)


#define num_vertices (1 << 6)

#define _U 0
#define _I 2

__global__ void transitive_closure_stage1_kernel(unsigned int* graph, int passnum)
{
    
    __shared__ unsigned int primary_block_buffer[BLOCK_DIM][BLOCK_DIM];

    int idxY = passnum * BLOCK_DIM + threadIdx.y;
    int idxX = passnum * BLOCK_DIM + threadIdx.x;

    primary_block_buffer[threadIdx.y][threadIdx.x] = _2D_ACCESS(graph, idxY, idxX, num_vertices);

    __syncthreads();

    for (unsigned int k = 0; k < BLOCK_DIM; ++k)
    {
        if ( primary_block_buffer[threadIdx.y][threadIdx.x] == _U)
        {
            if ( (primary_block_buffer[threadIdx.y][k] != _U) && (primary_block_buffer[k][threadIdx.x] != _U) )
            {
                primary_block_buffer[threadIdx.y][threadIdx.x] = passnum*BLOCK_DIM + k + _I;
            }
        }

        __syncthreads();
    }

    _2D_ACCESS(graph, idxY, idxX, num_vertices) = primary_block_buffer[threadIdx.y][threadIdx.x];
}

