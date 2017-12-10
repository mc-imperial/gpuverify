//pass
//--num_groups=[16,32] --local_size=[16,8]

__kernel void foo(__read_only image2d_t img, sampler_t samp)
{
    int2 idx = (int2)(get_global_id(0), get_global_id(1));
    float4 x = read_imagef(img, samp, idx);
}
