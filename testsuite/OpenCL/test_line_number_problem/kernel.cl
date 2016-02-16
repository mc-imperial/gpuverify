//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=64 --no-inline
//kernel.cl: error: possible write-read race on A\[[\d]+\]:[\s]
//Read by work item [\d]+ with local id [\d]+ in work group [\d]+.+kernel.cl:17:(14|16):[\s]+temp = A\[tid - i\];
//Write by work item[\s]+[\d]+ with local id [\d]+ in work group [\d]+.+kernel.cl:20:[\d]+:[\s]+A\[tid\] = A\[tid\] \+ temp;



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
