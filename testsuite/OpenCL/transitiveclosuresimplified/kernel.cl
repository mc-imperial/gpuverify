//pass
//--local_size=[8,8] --num_groups=[1,1] --equality-abstraction


#define _2D_ACCESS(A, y, x, X_DIM) A[(y)*(X_DIM)+(x)]

#define X_DIMENSION 0
#define Y_DIMENSION 1

#define LOCAL_SIZE (1 << 3)

#define num_vertices (1 << 6)

#define _U 0
#define _I 2

#define passnum 4

__kernel void transitive_closure_stage1_kernel(__global unsigned int* graph, __local unsigned int* primary_block_buffer)
{
    
    // Load primary block into shared memory (primary_block_buffer)

    // TODO: check that in OpenCL the order is 0=x, 1=y, 2=z (in AMP it is reversed)
    int idxY = passnum * LOCAL_SIZE + get_local_id(Y_DIMENSION);
    int idxX = passnum * LOCAL_SIZE + get_local_id(X_DIMENSION);

    _2D_ACCESS(primary_block_buffer, get_local_id(Y_DIMENSION), get_local_id(X_DIMENSION), LOCAL_SIZE) = _2D_ACCESS(graph, idxY, idxX, num_vertices);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k = 0; k < LOCAL_SIZE; ++k)
    {
        if ( _2D_ACCESS(primary_block_buffer, get_local_id(Y_DIMENSION), get_local_id(X_DIMENSION), LOCAL_SIZE) == _U)
        {
            if ( (_2D_ACCESS(primary_block_buffer, get_local_id(Y_DIMENSION), k, LOCAL_SIZE) != _U) && (_2D_ACCESS(primary_block_buffer, k, get_local_id(X_DIMENSION), LOCAL_SIZE) != _U) )
            {
                _2D_ACCESS(primary_block_buffer, get_local_id(Y_DIMENSION), get_local_id(X_DIMENSION), LOCAL_SIZE) = passnum*LOCAL_SIZE + k + _I;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    _2D_ACCESS(graph, idxY, idxX, num_vertices) = _2D_ACCESS(primary_block_buffer, get_local_id(Y_DIMENSION), get_local_id(X_DIMENSION), LOCAL_SIZE);
}

