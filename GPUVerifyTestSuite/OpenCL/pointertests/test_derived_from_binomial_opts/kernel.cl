//pass
//--local_size=64 --num_groups=64


__kernel void binomial_options_kernel(
                   const __global float* s, const __global float* x, 
                   const __global float* vdt, const __global float* pu_by_df, 
                   const __global float* pd_by_df,
                   __global float* call_value, 
                   __global float* call_buffer) 
{
    int tile_idx = get_group_id(0);
    int local_idx = get_local_id(0);

    __local float call_a[100+1];
    __local float call_b[100+1];


}

