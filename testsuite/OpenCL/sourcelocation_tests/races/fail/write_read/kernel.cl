//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//[\s]*kernel.cl:[\s]+error:[\s]+possible write-read race on \(\(char\*\)a\)\[12]
//kernel.cl:13:3:[\s]+read by thread[\s]+\([\d]+, 0, 0\)[\s]+group[\s]+\([\d]+, 0, 0\)[\s]+b\[get_local_id\(0\)] = a\[3];
//kernel.cl:11:5:[\s]+write by thread[\s]+\(3, 0, 0\)[\s]+group[\s]+\([\d]+, 0, 0\)[\s]+a\[get_local_id\(0\)] = get_local_id\(0\);



__kernel void foo(__local int* a, __local int* b) {

  a[get_local_id(0)] = get_local_id(0);
  
  b[get_local_id(0)] = a[3];
  
}