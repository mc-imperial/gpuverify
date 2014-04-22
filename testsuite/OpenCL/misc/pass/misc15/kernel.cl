//pass
//--local_size=2 --num_groups=1 --kernel-args=foo,0x00000003

kernel void foo (local int* A, int x) {
	if (x == 3) {
		A[get_local_id(0)] = 3;
	}
	else {
		A[0] = get_local_id(0);
	}
}
