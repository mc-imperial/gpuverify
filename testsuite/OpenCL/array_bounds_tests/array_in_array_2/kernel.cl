//pass
//--local_size=[5,8] --num_groups=1 --check-array-bounds

__kernel void foo() {
  int L[5] = {10, 20, 30, 40, 50};
  int K[40];
  K[get_global_id(1) * get_global_id(0)] = L[get_global_id(1) % 5];
}
