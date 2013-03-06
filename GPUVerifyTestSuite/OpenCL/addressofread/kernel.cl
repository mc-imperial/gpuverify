//pass
//--local_size=64 --num_groups=64



void baz(int* i)
{}

__kernel void foo() {
	int normal;
	int i;
	
	normal=i;
	
	
	baz(&i);
}


