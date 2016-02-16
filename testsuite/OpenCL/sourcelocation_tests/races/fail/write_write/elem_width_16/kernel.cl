//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible write-write race on q\[3]
//Write by work item[\s]+[\d]+[\s]+with local id[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:12:[\d]+:[\s]+q\[3] = get_local_id\(0\);




__kernel void foo(__global int * p, __global short * q) {
 
  p = (__global int *)q;
  q[3] = get_local_id(0);
}


