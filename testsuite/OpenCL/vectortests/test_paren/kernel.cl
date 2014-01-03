//pass
//--local_size=64 --num_groups=64 --equality-abstraction --no-inline



__kernel void foo(__global float4* acc) {

  float4 r;

  r = (float4)(1, 2, 3, 4);

  acc[get_local_id(0)] = r;

}
