//pass
//--local_size=1024 --num_groups=1024



__kernel void foo(__local int * p, __local int * q, __local int * r) {
 
  p[get_local_id(0)] = q[get_local_id(0)];
  r[get_local_id(0)] = q[get_local_id(0)];
  
}


