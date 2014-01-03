//pass
//--local_size=64 --num_groups=64 --no-inline


void bar(int * i)
{}

__kernel void foo() {
 
 int i = 0;
 i=0;
 
 bar(&i);
}


