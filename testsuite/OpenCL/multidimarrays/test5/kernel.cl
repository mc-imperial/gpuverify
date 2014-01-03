//pass
//--local_size=64 --num_groups=64 --equality-abstraction --no-inline
__kernel void foo(__global unsigned int* A)
{
    A[0] = 0;

    int x = A[0];
}

