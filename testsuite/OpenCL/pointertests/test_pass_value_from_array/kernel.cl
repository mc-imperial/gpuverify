//pass
//--local_size=64 --num_groups=64


void bar(float x)
{

}

__kernel void foo(__global float* A)
{
  bar(A[0]);
}

