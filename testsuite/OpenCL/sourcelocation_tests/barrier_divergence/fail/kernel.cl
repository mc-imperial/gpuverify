//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:13:5:[\s]+error:[\s]+barrier may be reached by non-uniform control flow[\s]+barrier\(CLK_GLOBAL_MEM_FENCE\);





__kernel void foo(__local int* a) {


  if (get_local_id(0) == 3) {
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  a[get_local_id(0)] = get_local_id(0);

}
