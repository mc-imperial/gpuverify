//pass
//--local_size=[64,64,64] --num_groups=[64,64,64] --no-inline



__kernel void foo() {

  __local float A[64][64][64];

  A[get_local_id(2)][get_local_id(1)][get_local_id(0)] = get_local_id(0);

}
