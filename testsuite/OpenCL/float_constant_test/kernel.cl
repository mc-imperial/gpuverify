//pass
//--local_size=[64,64] --num_groups=[64,64] --no-inline

float bar(void);

__kernel void foo()
{
    float f = bar();
    f = (float)2 - f;
}
