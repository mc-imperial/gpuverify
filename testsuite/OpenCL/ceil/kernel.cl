//pass
//--local_size=1024 --num_groups=1

__kernel void test(__global float4 *io)
{
    io[get_local_id(0)] = ceil(io[get_local_id(0)]);
}
