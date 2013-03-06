//pass
//--local_size=64 --num_groups=64


void bar(int x)
{
 int bar = x;
}

void baz(int y)
{
 int foo = y;
}

__kernel void foo(float x) {
  {
    int x = 4, y=2;
	x++;
	x++;
	y++;
	y++;
  }
  {
    int x = 2;
	int foo = 2;
  }
}
