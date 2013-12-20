//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible write-write race on \(\(char\*\)p\)\[[\d]+]
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d].+kernel.cl:9:5:[\s]+p\[get_local_id\(0\)] = get_group_id\(0\);



__kernel void foo(__global int* p) {
  p[get_local_id(0)] = get_group_id(0);
}
