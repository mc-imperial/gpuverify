//pass
//--local_size=64 --num_groups=64 --no-inline


void bar(int4 * p)
{
}

__kernel void foo() {
	
  int4 v;
  bar(&v);
	
}


