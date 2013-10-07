//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:[\s]+error:[\s]+possible write-write race on \(\(char\*\)p\)\[[\d]+\]
//kernel.cl:13:5:[\s]+write by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+[\s]+p\[get_global_id\(0\) \+ 1] = get_global_id\(0\);
//kernel.cl:9:5:[\s]+write by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+[\s]+p\[get_global_id\(0\)] = get_global_id\(0\);


__kernel void foo(__global int* p) {
  p[get_global_id(0)] = get_global_id(0);

  barrier(CLK_GLOBAL_MEM_FENCE);

  p[get_global_id(0) + 1] = get_global_id(0);
}
