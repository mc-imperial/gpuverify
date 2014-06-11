//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible write-write race on p\[[\d]+\]
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:13:5:[\s]+p\[get_global_id\(0\) \+ 1] = get_global_id\(0\);
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:9:5:[\s]+p\[get_global_id\(0\)] = get_global_id\(0\);


__kernel void foo(__global int* p) {
  p[get_global_id(0)] = get_global_id(0);

  barrier(CLK_GLOBAL_MEM_FENCE);

  p[get_global_id(0) + 1] = get_global_id(0);
}
