//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64 --no-inline
//Write by work item[\s]+[\d]+[\s]+with local id[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:9:[\d]+:[\s]+a = get_local_id\(0\);


__kernel void foo() {
  __local int a;

  a = get_local_id(0);

}

