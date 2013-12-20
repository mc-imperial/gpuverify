//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1 --no-inline
//B\[tid\] = v;
//v = atomic_add\(B\+i,v\);

// This is to test whether GPUVerify can correctly report the relevant atomic line
__kernel void blarp (global int* A, global int* B, global int* C, int x)
{
	int v;
	int i;

	int tid = get_global_id(0);

	for (i = 0; i < x; i++)
	{
		v = atomic_add(B+i,v);
	}

	B[tid] = v;
}
