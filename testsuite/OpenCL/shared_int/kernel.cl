//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//kernel.cl:9:7:[\s]+write by thread[\s]+\([\d]+, 0, 0\)[\s]+group[\s]+\([\d]+, 0, 0\)[\s]+a = get_local_id\(0\);


__kernel void foo() {
  __local int a;

  a = get_local_id(0);

}

