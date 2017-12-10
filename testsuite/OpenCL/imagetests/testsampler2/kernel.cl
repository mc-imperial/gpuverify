//pass
//--num_groups=[16,32] --local_size=[16,8]

const sampler_t samp = CLK_NORMALIZED_COORDS_TRUE
                     | CLK_ADDRESS_REPEAT
                     | CLK_FILTER_NEAREST;

__kernel void foo(__read_only image2d_t img)
{
    int2 idx = (int2)(get_global_id(0), get_global_id(1));
    float4 x = read_imagef(img, samp, idx);
}
