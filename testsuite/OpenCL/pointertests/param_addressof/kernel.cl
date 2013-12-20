//pass
//--local_size=64 --num_groups=64 --no-inline


void baz(int * a)
{
	*a = 1;
}

void bar()
{
	int * f;
	int g;
	
	f = &g;
	
	baz(&g);
	baz(f);
	
	if(g == 1 && *f == 1)
	{
	  //blah
	}
	
}

__kernel void foo(__global int* A, uint me)
{
	bar();
}

