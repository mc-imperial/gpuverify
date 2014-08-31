//xfail:COMMAND_LINE_ERROR
//--local_size=[32,64] --global_size=[64,64] --num_groups=[2,1]
//--num_groups=: not allowed with argument --global_size=

void __kernel a(__global int* A)
{
    A[get_global_id(0)] += 5;
}
