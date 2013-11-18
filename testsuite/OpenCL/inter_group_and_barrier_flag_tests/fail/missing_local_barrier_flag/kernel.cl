//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:[\s]+error:[\s]+possible[\s]+write-write[\s]+race on \(\(char\*\)p\)\[[\d]+]
//Write by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+.+kernel.cl:14:5:[\s]+p\[get_local_id\(0\) \+ 1] = get_global_id\(0\);
//Write by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+.+kernel.cl:10:5:[\s]+p\[get_local_id\(0\)] = get_global_id\(0\);


__kernel void foo(__local int* p) {

  p[get_local_id(0)] = get_global_id(0);

  barrier(CLK_GLOBAL_MEM_FENCE);

  p[get_local_id(0) + 1] = get_global_id(0);
}
