__kernel void foo(__global int *p) {
  p[get_local_id(0)] = get_local_id(0);
  p[get_local_id(0) + get_local_size(0) - 1] = get_local_id(0);
}
