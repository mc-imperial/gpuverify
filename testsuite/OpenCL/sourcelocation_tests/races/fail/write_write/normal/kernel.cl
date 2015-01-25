//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible write-write race on a\[7\]:
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:13:[\d]+:[\s]+a\[7\] = 0;
//Write by work item[\s]+7[\s]+in work group[\s]+[\d]+.+kernel.cl:11:[\d]+:[\s]+a\[get_local_id\(0\)\] = get_local_id\(0\);



__kernel void foo(__local int* a) {

  a[get_local_id(0)] = get_local_id(0);

  a[7] = 0;

}
