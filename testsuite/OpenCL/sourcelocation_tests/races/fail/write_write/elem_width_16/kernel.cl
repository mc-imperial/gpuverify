//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:[\s]+error:[\s]+possible write-write race on \(\(char\*\)q\)\[6]
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:12:10:[\s]+q\[3] = get_local_id\(0\);




__kernel void foo(__global int * p, __global short * q) {
 
  p = (__global int *)q;
  q[3] = get_local_id(0);
}


