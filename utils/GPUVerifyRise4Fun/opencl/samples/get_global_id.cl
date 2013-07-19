// --local_size=64 --num_groups=64

__kernel void foo(__global float* A) {
  A[get_global_id(0)] = A[get_global_id(0)] + 1.0f;
}
