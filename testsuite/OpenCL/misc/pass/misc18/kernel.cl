//PASS
//--local_size=2 --num_groups=2

#define LOCAL_SIZE 64
#define localId get_local_id(0)
#define groupId get_group_id(0)

kernel void reduction(global unsigned * __restrict input, global unsigned * __restrict output) {

	local float groupResults[LOCAL_SIZE];

	groupResults[localId] = input[localId];
  
        output[get_global_id(0)] = 0;

        __barrier_invariant_1(output[get_global_id(0)] == 0, get_local_id(0));
        barrier(CLK_LOCAL_MEM_FENCE);

}

