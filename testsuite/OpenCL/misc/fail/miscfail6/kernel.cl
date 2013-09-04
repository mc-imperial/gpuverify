//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1
//kernel.cl: error: possible read-write race

#define tid get_local_id(0)

__attribute__((always_inline)) inline void inlined(__local int *A, int offset)
{
   int temp = A[tid + offset];
   A[tid] += temp;
}

__kernel void inline_test(__local int *A, int offset) {
  inlined(A, offset);
}
