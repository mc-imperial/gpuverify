//pass
//--local_size=2 --global_size=4

constant sampler_t glb_smp = CLK_ADDRESS_CLAMP_TO_EDGE
                           | CLK_NORMALIZED_COORDS_TRUE
                           | CLK_FILTER_LINEAR;

void bar(sampler_t s);

kernel void foo(sampler_t smp_par) {

  sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE
                | CLK_NORMALIZED_COORDS_TRUE
                | CLK_FILTER_NEAREST;

  bar(smp);
  bar(glb_smp);
  bar(smp_par);
  bar(5);
}
