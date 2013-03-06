//pass
//--local_size=2048 --num_groups=4




__kernel void foo(__global int* p, __global int* q) {

  if(get_global_id(0) == 0) {
    p[4] = q[5];
    __assert(__implies(__write(p) & __read(q),
          __read_offset(q) == __write_offset(p) + sizeof(int*)));
  }
}