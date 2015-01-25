//xfail:NOT_ALL_VERIFIED
//--local_size=8 --num_groups=8 --check-array-bounds
//kernel.cl:8:[\d]+:[\s]+error:[\s]+possible array out-of-bounds access on array L

__kernel void foo() {
	local int L[20];
	int x = get_global_id(0);
	L[x] = x * x;
}
