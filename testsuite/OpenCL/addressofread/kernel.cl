//pass
//--local_size=64 --num_groups=64

int bar(void);

void baz(int* i)
{}

__kernel void foo() {
    int normal;
    int i = bar();

    normal=i;

    baz(&i);
}


