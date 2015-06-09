//pass
//--local_size=64 --num_groups=1 --equality-abstraction --no-inline

__kernel void foo(__local int* A, __local int* B, int i, int j) {
  __requires(i >= 0);
  __requires(j >= 0);
  A[i] = i;
  B[j] = A[j];
}
