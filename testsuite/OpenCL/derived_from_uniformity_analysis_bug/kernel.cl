//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo(__global int* A)
{
    for(int i = get_local_id(0); i < 100; i++) { }

    for(int j = 100; j > 0; j--) { }

}

