//pass
//--local_size=[8,8] --num_groups=8 --check-array-bounds

__kernel void foo() {
	local int L[8][8];
	L[get_local_id(0)][get_local_id(1)] = get_group_id(0);
}
