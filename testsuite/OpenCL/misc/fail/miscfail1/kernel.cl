//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1 --no-inline



#define tid get_local_id(0)
__kernel void foo(__local int* A, __local int* B, __local int *C) {
  C[tid] = C[tid + 1];
  B[tid] = A[C[tid + 2]];
}
