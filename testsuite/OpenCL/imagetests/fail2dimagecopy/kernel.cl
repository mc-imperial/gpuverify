//xfail:BOOGIE_ERROR
//--local_size=[8,8] --num_groups=[8,8]
//[\s]+possible





__kernel void k(__write_only image2d_t dest, image2d_t src) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  write_imagef(dest, (int2)(0, get_global_id(1)),
    read_imagef(src, sampler, (int2)(get_global_id(0), get_global_id(1))));
}
