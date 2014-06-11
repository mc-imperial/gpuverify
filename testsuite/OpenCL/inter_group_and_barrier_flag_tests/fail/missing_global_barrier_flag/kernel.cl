//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible write-write race on p\[[\d]+]
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d].+kernel.cl:15:7:[\s]+p\[get_global_id\(0\) \+ 1] = get_global_id\(0\);
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d].+kernel.cl:10:5:[\s]+p\[get_global_id\(0\)] = get_global_id\(0\);


__kernel void foo(__global int* p) {

  p[get_global_id(0)] = get_global_id(0);

  barrier(CLK_LOCAL_MEM_FENCE);

  if(get_local_id(0) < get_local_size(0) - 1) {
    p[get_global_id(0) + 1] = get_global_id(0);
  }
}
