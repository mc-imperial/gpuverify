//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:21:5:[\s]+read by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+[\s]+y = p\[0];
//kernel.cl:15:12:[\s]+write by thread[\s]+[\d]+[\s]+in group[\s]+[\d]+[\s]+p\[0] = get_local_id\(0\);



__kernel void foo(__local int* p) {

  volatile int x, y;

  x = get_local_id(0) == 0 ? 0 : CLK_LOCAL_MEM_FENCE;

  if(get_local_id(0) == 0) {
    p[0] = get_local_id(0);
  }

  barrier(x);  

  if(get_local_id(0) == 1) {
    y = p[0];
  }

}
