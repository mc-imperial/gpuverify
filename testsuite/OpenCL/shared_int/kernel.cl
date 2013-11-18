//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//Write by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+.+kernel.cl:9:7:[\s]+a = get_local_id\(0\);


__kernel void foo() {
  __local int a;

  a = get_local_id(0);

}

