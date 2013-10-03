//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:[\s]+error:[\s]+possible write-write race on \(\(char\*\)q\)\[6]
//kernel.cl:12:10:[\s]+write by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+[\s]+q\[3] = get_local_id\(0\);




__kernel void foo(__global int * p, __global short * q) {
 
  p = (__global int *)q;
  q[3] = get_local_id(0);
}


