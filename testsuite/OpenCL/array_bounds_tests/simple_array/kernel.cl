//pass
//--local_size=8 --num_groups=8 --check-array-bounds

__kernel void foo() {
	local int L[64];
	L[get_global_id(0)] = get_global_size(0);
}
