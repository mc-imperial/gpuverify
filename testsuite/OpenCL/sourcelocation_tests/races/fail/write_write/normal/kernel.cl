//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:[\s]+error:[\s]+possible write-write race on \(\(char\*\)a\)\[28\]:
//kernel.cl:13:3:[\s]+write by thread[\s]+\([\d]+, 0, 0\)[\s]+group[\s]+\([\d]+, 0, 0\)[\s]+a\[7\] = 0;
//kernel.cl:11:5:[\s]+write by thread[\s]+\(7, 0, 0\)[\s]+group[\s]+\([\d]+, 0, 0\)[\s]+a\[get_local_id\(0\)\] = get_local_id\(0\);



__kernel void foo(__local int* a) {

  a[get_local_id(0)] = get_local_id(0);
  
  a[7] = 0;
  
}
