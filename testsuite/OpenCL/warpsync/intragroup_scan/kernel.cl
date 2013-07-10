//pass
//--local_size=1024 --num_groups=1 --warp-sync=32

inline void scan_warp (local int* A)
{
	unsigned int tid = get_local_id(0);
	unsigned int lane = tid % 32;

	if (lane >= 1) A[tid] = A[tid - 1] + A[tid];
	if (lane >= 2) A[tid] = A[tid - 2] + A[tid];
	if (lane >= 4) A[tid] = A[tid - 4] + A[tid];
	if (lane >= 8) A[tid] = A[tid - 8] + A[tid];
	if (lane >= 16) A[tid] = A[tid - 16] + A[tid];
}

__kernel void scan (local int* A)
{
	unsigned int tid = get_local_id(0);
	unsigned int lane = tid % 32;
	local int temp [32];
	scan_warp(A);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lane == 31)
		temp[tid / 32] = A[tid];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid / 32 == 0)
		scan_warp(temp);
	barrier(CLK_LOCAL_MEM_FENCE);
	A[tid] += temp[tid/32];
}
