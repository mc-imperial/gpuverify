// --local_size=64 --num_groups=64
__kernel void foo(__local int* A, __local int* B, int i, int j) {
  A[i] = i;
  B[j] = A[j];
}
