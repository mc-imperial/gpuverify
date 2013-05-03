//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=64
//kernel.cl: error: possible write-read race on \(\(char\*\)A\)\[[\d]+\]:[\s]
//kernel.cl:17:16:[\s]read by thread \([\d]+, [\d]+, [\d]+\) group \([\d]+, [\d]+, [\d]+\)[\s]+temp = A\[tid - i\];
//kernel.cl:20:9:[\s]+write by thread[\s]+\([\d]+, [\d]+, [\d]+\) group \([\d]+, [\d]+, [\d]+\)[\s]+A\[tid\] = A\[tid\] \+ temp;



#define sz get_local_size(0)
#define tid get_local_id(0)

__kernel void foo(__local int *A) {
  int temp;
  int i = 1;
  while(i < sz) {
    if(i < tid)
      temp = A[tid - i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i < tid)
      A[tid] = A[tid] + temp;
    i = i * 2;
  }
}
