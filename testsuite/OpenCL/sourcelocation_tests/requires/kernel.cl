//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:18:7:[\s]+error: a precondition for this call might not hold[\s]+x = bar\(0\);[\s]
//kernel.cl:10:24:[\s]+note:[\s]+this is the precondition that might not hold[\s]+__requires\(__implies\(__enabled\(\), a > 0\)\);




int bar(int a) {
  __requires(__implies(__enabled(), a > 0));
  __ensures(__implies(__enabled(), __return_val_int() >= 0));
  return a/2;
}

__kernel void foo() {

  int x;
  x = bar(0);

}

