//pass
//--local_size=2048 --num_groups=1024 --no-inline



__kernel void foo(__global int* p) {
  p[get_global_id(0)] = get_local_id(0);
}
