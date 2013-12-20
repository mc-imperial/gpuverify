//pass
//--local_size=128 --num_groups=128 --no-inline



__kernel void foo(__local int* p, __local int* q) {

  p[get_local_id(0)] = q[get_local_id(0)];

  __assert(!__write(q));

}
