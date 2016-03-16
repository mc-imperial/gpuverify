//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//[\s]*kernel.cl:[\s]+error:[\s]+possible read-write race on a\[8]
//Write by work item[\s]+[\d]+[\s]+with local id[\s]+8+[\s]+in work group[\s]+[\d]+.+kernel.cl:13:[\d]+:[\s]+a\[get_local_id\(0\)] = get_local_id\(0\);
//Read by work item[\s]+[\d]+[\s]+with local id[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:11:(24|3):[\s]+b\[get_local_id\(0\)] = a\[8];



__kernel void foo(__local int* a, __local int* b) {

  b[get_local_id(0)] = a[8];

  a[get_local_id(0)] = get_local_id(0);

}
