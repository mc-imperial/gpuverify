//pass
//--local_size=1024 --num_groups=1024



__kernel void foo(__local int* a, __local int* b) {

  a[get_local_id(0)] = get_local_id(0);
  
  barrier(CLK_GLOBAL_MEM_FENCE);

  b[get_local_id(0)] = get_local_id(0);

}