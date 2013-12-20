//pass
//--local_size=1024 --num_groups=1024 --no-inline

void f2(__global int *a)
{
}

void f1(__global int *a){
  f2(a);
}

__kernel
void k(__global int *a)
{
    if (get_local_id(0)) {
        f1(a);
    }
}
