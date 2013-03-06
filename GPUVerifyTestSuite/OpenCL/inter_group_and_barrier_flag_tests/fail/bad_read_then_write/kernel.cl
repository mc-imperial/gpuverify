//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:[\s]+error:[\s]+possible[\s]+read-write[\s]+race on \(\(char\*\)p\)\[0]
//kernel.cl:19:5:[\s]+write by thread[\s]+\([\d]+, 0, 0\)[\s]+group[\s]+\([\d]+, 0, 0\)[\s]+p\[0] = 0;
//kernel.cl:14:3:[\s]+read by thread[\s]+\([\d]+, 0, 0\)[\s]+group[\s]+\([\d]+, 0, 0\)[\s]+y = p\[0];


__kernel void foo(__local int* p) {

  volatile int x, y;

  x = get_local_id(0) == 0 ? CLK_LOCAL_MEM_FENCE : 0;

  y = p[0]; // kernel.cl:19:5:

  barrier(x);  

  if(get_local_id(0) == 1) {
    p[0] = 0;
  }

}
