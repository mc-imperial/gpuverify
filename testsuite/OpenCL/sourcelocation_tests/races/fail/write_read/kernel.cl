//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//[\s]*kernel.cl:[\s]+error:[\s]+possible write-read race on a\[3]
//Read by work item[\s]+[\d]+[\s]+with local id[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:13:(24|3):[\s]+b\[get_local_id\(0\)] = a\[3];
//Write by work item[\s]+[\d]+[\s]+with local id[\s]+3[\s]+in work group[\s]+[\d]+.+kernel.cl:11:[\d]+:[\s]+a\[get_local_id\(0\)] = get_local_id\(0\);



__kernel void foo(__local int* a, __local int* b) {

  a[get_local_id(0)] = get_local_id(0);

  b[get_local_id(0)] = a[3];

}
