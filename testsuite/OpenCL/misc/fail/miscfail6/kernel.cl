//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1
//kernel.cl: error: possible read-write race
//GPUVerify kernel analyser finished with 0 verified, 1 error

// The always_inline attribute ensures that the 'inlined' function is actually
// inlined in 'inline_test'. The inline keyword ensures that the body of
// 'inlined' is not kept around as a seperate function in the llvm bitcode.
// Hence, we should get exactly 1 error.

#define tid get_local_id(0)

__attribute__((always_inline)) inline void inlined(__local int *A, int offset)
{
   int temp = A[tid + offset];
   A[tid] += temp;
}

__kernel void inline_test(__local int *A, int offset) {
  inlined(A, offset);
}
