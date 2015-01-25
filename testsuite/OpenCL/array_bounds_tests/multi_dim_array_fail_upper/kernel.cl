//xfail:NOT_ALL_VERIFIED
//--local_size=[10,10,10] --num_groups=2 --check-array-bounds
//kernel.cl:7:[\d]+:[\s]+error:[\s]+possible array out-of-bounds access on array L

__kernel void foo() {
	local int L[10][10][10];
	L[get_global_id(0)][get_local_id(1)][get_local_id(2)] = get_group_id(0);
}
