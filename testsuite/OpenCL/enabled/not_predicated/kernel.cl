//pass
//--local_size=16 --num_groups=1 --no-inline

__kernel void foo(__global int* A) {
  __assert(__enabled());
}
