//pass
//--blockDim=1024 --gridDim=1

__device__ void bar(char **in, char **out) {
  char tmp = (*in)[threadIdx.x];
  out[0][threadIdx.x] = tmp;
  *out = *in;
}

__global__ void foo(char *A, char *B, char c)
{
  char *choice1 = c ? A : B;
  char *choice2 = c ? B : A;
  bar(&choice1, &choice2);
  bar(&choice1, &choice2);
}
