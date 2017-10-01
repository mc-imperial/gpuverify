//pass
//--local_size=512,512 --num_groups=256,256

__kernel void test() {
  __assert(get_work_dim() == 2);
}
