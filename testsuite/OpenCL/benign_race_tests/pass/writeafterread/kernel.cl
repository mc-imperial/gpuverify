//pass
//--local_size=64 --num_groups=1 --equality-abstraction --no-inline

__kernel void foo(__local int* A) {
  int v;

  v = A[0];
  A[0] = v;
}
