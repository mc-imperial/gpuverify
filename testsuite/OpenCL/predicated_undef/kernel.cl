//pass
//--local_size=64 --num_groups=64

__kernel void k(__global int *input, __global int *output) {

  __local int sum[1024];
  int temp;
  int offset = 1;

    if (get_local_id(0) >= offset)
    {
      // Purpose: to test uniformity analysis with undefined.
      sum[get_local_id(0)] = temp;
    }


}
