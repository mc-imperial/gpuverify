//xfail:COMMAND_LINE_ERROR
//--local_size=[32,64] --global_size=[64,64] --num_groups=[2,1]
//--num_groups and --global_size are mutually exclusive.

void __kernel a(__global int* A)
{
    A[get_global_id(0)] += 5;
}
