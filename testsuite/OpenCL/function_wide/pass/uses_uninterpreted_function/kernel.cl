//pass
//--no-infer --local_size=1024 --num_groups=2

bool __uninterpreted_function_bar(void);

kernel void foo() {
    __function_wide_invariant(__uninterpreted_function_bar());
}
