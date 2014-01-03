//pass
//--local_size=64 --num_groups=64 --no-inline



__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void foo(__read_only image2d_t matrixA) {

  float4 tempA0 = read_imagef(matrixA, imageSampler, (int2)(1));

}
