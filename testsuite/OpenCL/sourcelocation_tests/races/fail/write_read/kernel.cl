//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//[\s]*kernel.cl:[\s]+error:[\s]+possible write-read race on \(\(char\*\)a\)\[12]
//Read by work item[\s]+[\d]+[\s]+in work group[\s]+[\d]+.+kernel.cl:13:3:[\s]+b\[get_local_id\(0\)] = a\[3];
//Write by work item[\s]+3[\s]+in work group[\s]+[\d]+.+kernel.cl:11:5:[\s]+a\[get_local_id\(0\)] = get_local_id\(0\);



__kernel void foo(__local int* a, __local int* b) {

  a[get_local_id(0)] = get_local_id(0);
  
  b[get_local_id(0)] = a[3];
  
}
