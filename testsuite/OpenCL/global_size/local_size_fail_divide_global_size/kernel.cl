//xfail:COMMAND_LINE_ERROR
//--local_size=[32,64] --global_size=[64,127]
//Dimension 1 of global size does not divide by dimension 1 of local size

void __kernel a(__global int* A)
{
    A[get_global_id(0)] += 5;
}
