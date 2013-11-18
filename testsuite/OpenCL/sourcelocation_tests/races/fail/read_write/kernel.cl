//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//[\s]*kernel.cl:[\s]+error:[\s]+possible read-write race on \(\(char\*\)a\)\[32]
//Write by thread[\s]+8[\s]+in group[\s]+[\d]+.+kernel.cl:13:5:[\s]+a\[get_local_id\(0\)] = get_local_id\(0\);
//Read by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+.+kernel.cl:11:3:[\s]+b\[get_local_id\(0\)] = a\[8];



__kernel void foo(__local int* a, __local int* b) {

  b[get_local_id(0)] = a[8];
  
  a[get_local_id(0)] = get_local_id(0);
  
}
