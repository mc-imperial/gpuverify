//pass
//--local_size=64 --num_groups=64


__kernel void foo(__local float* A) {

  A[get_local_id(0) == 0 ? 1 : 2*get_local_id(0)] = 2.4f;

}
