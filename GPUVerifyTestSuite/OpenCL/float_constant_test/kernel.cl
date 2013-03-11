//pass
//--local_size=[64,64] --num_groups=[64,64]

__kernel void foo()
{
    float f;
    f = (float)2 - f;
    
}
