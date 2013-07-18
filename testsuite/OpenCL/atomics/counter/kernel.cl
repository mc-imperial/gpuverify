//pass
//--local_size=1024 --num_groups=1

// This currently fails to verify because, while no two returns from atomic_inc will be the same, GPUVerify can't infer that
kernel void counter (local int* A)
{
	local int count;
	A[atomic_inc(&count)] = get_local_id(0);
}
