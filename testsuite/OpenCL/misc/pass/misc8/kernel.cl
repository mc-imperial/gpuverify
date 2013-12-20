//pass
//--local_size=1024 --num_groups=1024 --no-inline

float baz(float p)
{
    return half_powr(2.0f, p);
}

__kernel void foo(__global float *A)
{
    if(A[get_global_id(0)])
      baz(2.0f);
}
