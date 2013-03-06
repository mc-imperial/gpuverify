//pass
//--local_size=64 --num_groups=64


void bar(int p)
{
  __ensures(p == __old_int(p));

}

__kernel void foo() {

  bar(4);

}