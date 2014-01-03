//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=32 --no-inline

#define MAX 1024

void notInvoked(__local float* dst) {
  int i = get_local_id(0);
  while(i < MAX) {
    dst[i] = 0;
    i += get_local_size(0);
  }
}

__kernel void foo() {
    // deliberately does not invoke anything
}

