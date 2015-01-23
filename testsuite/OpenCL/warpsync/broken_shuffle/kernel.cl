//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1 --warp-sync=16 --no-inline

__kernel void shuffle (__local int* A)
{
	int tid = get_local_id(0);
	int warp = tid / 32;
	__local int* B = A + (warp*32);
	A[tid] = B[(tid + 1)%32];
}
