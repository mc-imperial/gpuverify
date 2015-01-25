//xfail:NOT_ALL_VERIFIED
//--local_size=8 --num_groups=8 --check-array-bounds
//kernel.cl:7:[\d]+:[\s]+error:[\s]+possible array out-of-bounds access on array L

__kernel void foo() {
	local int L[64];
	L[0 - get_global_id(0)] = get_global_size(0);
}
