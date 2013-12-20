//pass
//--local_size=512 --num_groups=256 --no-inline



__kernel void foo(__global float* A) {
  A[get_global_id(0)] = A[get_global_id(0)] + 1.0f;
}
