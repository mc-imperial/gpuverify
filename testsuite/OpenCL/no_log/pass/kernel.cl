//pass
//--local_size=16 --num_groups=1 --only-log --no-inline

__kernel void foo(__local int* A) {
  // Causes a race, but with --only-log this is not reported
  A[0] = get_local_id(0);
}
