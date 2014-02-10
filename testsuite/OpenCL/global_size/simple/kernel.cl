//pass
//--local_size=[32,64] --global_size=[64,256]

void __kernel a(__global int* A)
{
    __assert( get_global_size(0) == 64);
    __assert( get_global_size(1) == 256);
    __assert( get_num_groups(0) == 2);
    __assert( get_num_groups(1) == 4);
    A[get_global_id(0) + get_global_id(1)*get_global_size(0)] +=5;
}
