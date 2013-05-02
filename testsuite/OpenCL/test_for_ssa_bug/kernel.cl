//pass
//--local_size=2 --num_groups=1




__kernel void foo(__local unsigned int* A)
{
    for (unsigned int k = 0; k < 8; ++k)
    {
        if (get_local_id(0) == 0)
        {
            A[get_local_id(0)] = 0 + k;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

} 
