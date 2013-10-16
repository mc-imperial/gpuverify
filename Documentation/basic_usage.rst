========================
Basic Usage
========================


Running GPUVerify
=================

OpenCL
------

To invoke GPUVerify on an OpenCL kernel, do::

  gpuverify --local_size=<work-group-dimensions> --num_groups=<grid-dimensions> <OpenCL file> 

Here, ``<work-group-dimensions>`` is a vector specifying the dimensionality of each work group, ``<grid-dimensions>`` is a vector specifying the dimensionality of the grid of work groups, and ``<OpenCL file>`` is an OpenCL file with extension ``.cl``.

For example, to check ``kernel.cl`` with respect to a (16x16) grid of work groups, each of which is a (32x32) arrangement of work-items, do::

  gpuverify --num_groups=[16,16] --local_size=[32,32] kernel.cl

For brevity, a 1D vector ``[x]`` can be written as simply ``x``. So,
for instance, to check ``kernel.cl`` with respect to a single work
group which is a (32x32) arrangement of work-items, you can specify::

  gpuverify --num_groups=1 --local_size=[32,32] kernel.cl 

which is equivalent to::

  gpuverify --num_groups=[1] --local_size=[32,32] kernel.cl 


CUDA
----

To invoke GPUVerify on a CUDA kernel, do::

  gpuverify --blockDim=<block-dimensions> --gridDim=<grid-dimensions> <CUDA file>

Here, ``<block-dimensions>`` is a vector specifying the dimensionality of each thread block, ``<grid-dimensions>`` is a vector specifying the dimensionality of the grid of thread blocks, and ``<CUDA file>`` is a CUDA file with extension ``.cu``.

For example, to check ``kernel.cu`` with respect to a (16x16) grid of
thread blocks, each of which is a (32x32) arrangement of threads, do::

  gpuverify --gridDim=[16,16] --blockDim=[32,32] kernel.cu

For brevity, a 1D vector ``[x]`` can be written as simply ``x``. So,
for instance, to check ``kernel.cl`` with respect to a single work
group which is a (32x32) arrangement of work-items, you can specify::

  gpuverify --gridDim=1 --blockDim=[32,32] kernel.cu

which is equivalent to::

  gpuverify --gridDim=[1] --blockDim=[32,32] kernel.cu 


Usage modes
===========

There are two main usage modes:

* Verify mode (default)
* Findbugs mode, specified via the ``--findbugs`` flag

Verify mode
-----------

By default, GPUVerify runs in verify mode. In this mode, the tool will
attempt to verify that the input kernel is free from defects. If
verification succeeds, the tool will report that the kernel has been
verified as free from the types of defects which GPUVerify can check
for. This verification result can be trusted, modulo bugs in GPUVerify
and known sources of unsoundness in the tool.

.. todo:: In due course this
  documentation will be updated to describe potential sources of
  unsoundness.

In verify mode, any defects reported by GPUVerify are **possible**
defects.  If the kernel contains loops there is a high chance that
they may be *false positives* arising due to limitations of the
tool's invariant inference procedures.

A similar issue as with loop invariants should not arise when multiple
(non-inlined) procedures are present; procedures are automatically inlined.
This does mean that the tool will report an error when multual recursive
procedures are present.  To verify multual recursive procedures, disable
automatic inlining with the ``--no-inline`` flag.  Observe that this will
increase the chance of *false positives* arising due to limitations of the
tool's contract inference procedures.

Findbugs mode
-------------

The ``--findbugs`` flag causes GPUVerify to run in *findbugs*
mode. In this mode, all loops are unwound by a fixed (default=2)
number of iterations; executions that go beyond this unwinding depth
are not checked.

If GPUVerify reports that no defects were found when running in
findbugs mode this does not mean that the kernel is free from
defects.  If defects are reported, they are much more
likely to be genuine than when the tool runs in verify mode, because
loops are not abstracted using invariants. However, GPUVerify can
still report false positives because:

* shared arrays are modelled abstractly
* floating point calculations are represented abstractly (so that, for
  example, if ``x`` is a ``float`` variable, GPUVerify does
  not know that (assuming ``x`` is not a NaN) ``x <= x + 1.0f`` holds)


Properties checked by GPUVerify
===============================

We now describe the key properties of GPU kernels that GPUVerify
checks, giving a mixture of small OpenCL and CUDA kernels as examples.

Intra-group data races
----------------------

* OpenCL: an intra-group data race is a race between work items
  in the same work group.
* CUDA: an intra-group data race is a
  race between threads in the same thread block.

Suppose the following OpenCL kernel is executed by a single work group
consisting of 1024 work items::

  1  __kernel void foo(__global int *p) {
  2    p[get_local_id(0)] = get_local_id(0);
  3    p[get_local_id(0) + get_local_size(0) - 1] = get_local_id(0);
  4  }

An intra-group data race can occur between work items 0 and 1023. If
we run GPUVerify on the example::

  gpuverify --local_size=1024 --num_groups=1 intra-group.cl

then this intra-group race is detected::

  intra-group.cl: error: possible write-write race on ((char*)p)[4092]:

  intra-group.cl:3:23: write by thread (0, 0, 0) group (0, 0, 0)
   p[get_local_id(0) + get_local_size(0) - 1] = get_local_id(0);

  intra-group.cl:2:5:  write by thread (1023, 0, 0) group (0, 0, 0)
   p[get_local_id(0)] = get_local_id(0);

Inter-group data races
----------------------

* OpenCL: an inter-group data race is a race between work items in
  different work groups.
* CUDA: an inter-group data race is a race between threads in different thread blocks.

Suppose the following CUDA kernel is executed by 8 thread blocks each
consisting of 64 work items::

  1  #include <cuda.h>
  2  
  3  __global__ void foo(int *p) {
  4    p[threadIdx.x] = threadIdx.x;
  5  }

The kernel is free from intra-group data races, but inter-group data
race can occur between threads in different blocks that have identical
intra-block thread indices. If we run GPUVerify on the example::

  gpuverify --blockDim=64 --gridDim=8 inter-group.cu

then an inter-group race is detected::

  inter-group.cu: error: possible write-write race on ((char*)p)[0]:

  inter-group.cu:4:3: write by thread (0, 0, 0) group (0, 0, 0)
   p[threadIdx.x] = threadIdx.x;

  inter-group.cu:4:3: write by thread (0, 0, 0) group (1, 0, 0)
   p[threadIdx.x] = threadIdx.x;


Barrier divergence
------------------

GPUVerify detects cases where a kernel breaks the rules for barrier synchronization in conditional code defined in the CUDA and OpenCL documentation. In particular, the tool checks that if a barrier occurs in a conditional statement then all threads must evaluate the condition uniformly, and if a barrier occurs inside a loop then all threads must execute the same number of loop iterations before synchronizing at the barrier.

GPUVerify rejects the following OpenCL kernel, executed by a single
work group of 1024 work items, because work items will execute
different numbers of loop iterations, breaking the barrier
synchronization rules::

  1  __kernel void foo(__global int *p) {
  2    for(int i = 0; i < get_global_id(0); i++) {
  3      p[i + get_global_id(0)] = get_global_id(0);
  4      barrier(CLK_GLOBAL_MEM_FENCE);
  5    }
  6  }

::

  gpuverify --local_size=1024 --num_groups=1 barrier-div-opencl.cl

::

  barrier-div.cl:4:5: error: barrier may be reached by non-uniform control flow
     barrier(CLK_GLOBAL_MEM_FENCE);

GPUVerify rejects the following CUDA kernel when, say, executed by a
32x32 grid of 16x16 thread blocks. The reason is that the tool assumes
the contents of array p are arbitrary, so there is no guarantee that
all threads in a thread block will reach the same barrier::

  1  #include <cuda.h>
  2  
  3  __global__ void foo(int *p) {
  4    if(p[threadIdx.x]) {
  5      // May be reached by some threads but not others depending on contents of p
  6      __syncthreads();
  7    }  
  8  }

::

  gpuverify --blockDim=[16,16] --gridDim=[32,32] barrier-div-cuda.cu

::

  barrier-div-cuda.cu:6:5: error: barrier may be reached by non-uniform control flow
     __syncthreads(); 


Command Line Options
====================

.. todo:: Get these from GPUVerify and expand on them.
