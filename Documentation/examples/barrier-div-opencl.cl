__kernel void foo(__global int *p) {
  for(int i = 0; i < get_global_id(0); i++) {
    p[i + get_global_id(0)] = get_global_id(0);
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}