//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=24 --no-infer --k-induction-depth=2

__kernel void foo(int a, int b, int c) {
    __requires(a != b);
    __requires(a != c);
    __requires(b != c);

    for(int i = 0; __invariant(a != b), i < 100; i++) {
        int temp = a;
        a = b;
        b = c;
        c = temp;
    }
    __assert(a != b);
}
