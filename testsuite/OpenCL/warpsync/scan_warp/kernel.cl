//pass
//--local_size=1024 --num_groups=1 --warp-sync=32 --no-inline

__kernel void scan (local int* A)
{
	int tid = get_local_id(0);
	unsigned int lane = tid & 31;

	if (lane >= 1) A[tid] = A[tid - 1] + A[tid];
	if (lane >= 2) A[tid] = A[tid - 2] + A[tid];
	if (lane >= 4) A[tid] = A[tid - 4] + A[tid];
	if (lane >= 8) A[tid] = A[tid - 8] + A[tid];
	if (lane >= 16) A[tid] = A[tid - 16] + A[tid];
}
