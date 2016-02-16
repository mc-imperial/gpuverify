//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible write-write race on a\[7\]:
//Write by work item[\s]+[\d]+[\s]+with local id 0 in work group[\s]+[\d]+.+kernel.cl:13:[\d]+:[\s]+a\[7\] = 0;

//Write by work item[\s]+[\d]+[\s]+with local id[\s]+7[\s]+in work group[\s]+[\d]+.+kernel.cl:[\d]+:[\d]+:[\s]+a\[get_local_id\(0\)\] = get_local_id\(0\);


__kernel void foo(__local int* a) {

  a[get_local_id(0)] = get_local_id(0);
  if(get_local_id(0) == 0)
    a[7] = 0;

}


