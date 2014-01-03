//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible[\s]+read-write[\s]+race on \(\(char\*\)p\)\[0]
//Write by work item[\s]+[\d]+[\s]+in work group[\s]+[\d].+kernel.cl:19:5:[\s]+p\[0] = 0;
//Read by work item[\s]+[\d]+[\s]+in work group[\s]+[\d].+kernel.cl:14:3:[\s]+y = p\[0];


__kernel void foo(__local int* p) {

  volatile int x, y;

  x = get_local_id(0) == 0 ? CLK_LOCAL_MEM_FENCE : 0;

  y = p[0]; // kernel.cl:19:5:

  barrier(x);  

  if(get_local_id(0) == 1) {
    p[0] = 0;
  }

}
