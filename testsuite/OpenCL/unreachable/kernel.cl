//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1
//this assertion might not hold for work item
void f(int *a) __attribute__ ((noreturn));

__kernel
void k(int *a)
{
  f(a);
}
