//xfail:COMMAND_LINE_ERROR
//--local_size=[32,64] --global_size=[64]
//Dimensions of local and global size must match

void __kernel a(__global int* A)
{
    A[get_global_id(0)] += 5;
}
