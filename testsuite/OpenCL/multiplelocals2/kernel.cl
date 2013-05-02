//pass
//--local_size=64 --num_groups=64

__kernel void foo()
{
    for(int i=0; i < 2; i++)
	{
	  i++;
	}
	
	for(int i=0; i < 2; i++)
	{
	  i++;
	}
	
}
