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

An alternative to specifying the grid of workgroups using ``--num-groups=`` is
to use the ``--global_size=`` argument instead. This allows the size of the
global NDRange to be specified instead of the number of workgroups. The
``--global_size=`` and ``--local_size`` arguments have a direct correspondence
to the ``global_work_size`` and ``local_work_size`` parameters to the OpenCL
runtime function ``clEnqueueNDRangeKernel()`` respectively.

Here is an example using ``--global_size=``::

  gpuverify --local_size=[32,32] --global_size=[512,512] kernel.cl

Which is equivalent to the following using ``--num_groups=`` instead of ``--global_size=``::

  gpuverify --local_size=[32,32] --num_groups=[16,16] kernel.cl

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

.. _verifymode:

Verify mode
-----------

By default, GPUVerify runs in verify mode. In this mode, the tool will
attempt to verify that the input kernel is free from defects. If
verification succeeds, the tool will report that the kernel has been
verified as free from the types of defects which GPUVerify can check
for. This verification result can be trusted, modulo bugs in GPUVerify
and known sources of unsoundness in the tool, see :doc:`limitations`.

In verify mode, any defects reported by GPUVerify are **possible**
defects.  If the kernel contains loops there is a high chance that
they may be *false positives* arising due to limitations of the
tool's invariant inference procedures.

A similar issue as with loop invariants should not arise when multiple
(non-inlined) procedures are present; procedures are automatically inlined.
This does mean that the tool will report an error when mutual recursive
procedures are present.  To verify mutual recursive procedures, disable
automatic inlining with the ``--no-inline`` flag.  Observe that this will
increase the chance of *false positives* arising due to limitations of the
tool's contract inference procedures.


.. _findbugs:

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

* shared arrays are modeled abstractly
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

In the description of command line options, we follow OpenCL terminology, not CUDA terminology.  We thus refer to work items and work groups, not threads and thread blocks, and to local memory, not shared memory.

General options
---------------

-h, --help
^^^^^^^^^^

Display list of GPUVerify options.  Please report cases where GPUVerify claims to have an option not documented here, or if an option mentioned here is not listed by GPUVerify.

``--version``
^^^^^^^^^^^^^

Show version information.

-D <value>
^^^^^^^^^^

Define symbol

-I <value>
^^^^^^^^^^

Add directory to include search path

``--findbugs``
^^^^^^^^^^^^^^

Run tool in bug-finding mode, see :ref:`findbugs`.  In this mode, loop invariant inference is disabled, and a loop unwinding depth of 2 is used, unless this depth is over-ridden using ``--loop-unwind``.

``--verify``
^^^^^^^^^^^^

Run GPUVerify in *verify* mode (see :ref:`verifymode`).  This is the mode the tool uses by default.

``--loop-unwind=``\X
^^^^^^^^^^^^^^^^^^^^

Run tool in *findbugs* mode (see :ref:`findbugs`) and explore only traces that
pass through at most X loop heads.

``--no-benign-tolerance``
^^^^^^^^^^^^^^^^^^^^^^^^^

By default, GPUVerify tries to tolerate certain kinds of (arguably) *benign* data races.  For example, if GPUVerify can figure out that in a write-write data race, both work items involved are guaranteed to write the same value to the memory location in question, it will not report the race.

Sometimes we wish to turn off this tolerance, perhaps because we believe our kernel should be free from such races, or because we are feeling strict and want to take the (arguably correct) view that "all data races are evil with no exceptions".

.. todo:: Add link to the paper with this title.

Also, it may be the case (though we have not evaluated this systematically) that tolerating benign races carries some performance overhead in terms of verification time.

To disable tolerance of benign races, specify ``--no-benign-tolerance``.

``--only-divergence``
^^^^^^^^^^^^^^^^^^^^^

Disable race checking, and only check for barrier divergence.

``--only-intra-group``
^^^^^^^^^^^^^^^^^^^^^^

Do not check for inter-work-group races.  In this mode, a kernel may be deemed correct even if it can exhibit races on global memory between work items in different work groups, as long as GPUVerify can prove that there are no data races (on global or local memory) between work items in the same work group.

.. _verbose:

``--verbose``
^^^^^^^^^^^^^

With this option, GPUVerify will print the various sub-commands that are issued during the analysis process.  Also, output produced by the tools which GPUVerify invokes will be displayed.  If you are debugging, and are issuing print statements in one of the GPUVerify components, you will need to use ``--verbose`` to be able to see the results of this printing.

``--silent``
^^^^^^^^^^^^

Silent on success; only show errors/timing

``--time``
^^^^^^^^^^

When GPUVerify finishes, print statistics about timing.

``--time-as-csv=``\X
^^^^^^^^^^^^^^^^^^^^

Print timing data as CSV row with label X.

``--timeout=``\X
^^^^^^^^^^^^^^^^

Allow each component to run for X seconds before giving up.  Specifying 0 disables the timeout. The default is 300 seconds.

``--opencl``
^^^^^^^^^^^^

Assume the kernel to verify is an OpenCL kernel. By default GPUVerify tries to detect whether the kernel to be verified is an OpenCL or CUDA kernel based on file extension and file contents. When detection fails, the kernel type can be explicitly specified to be OpenCL by passing this option.

``--cuda``
^^^^^^^^^^

Assume the kernel to verify is a CUDA kernel. Similar to the ``--opencl`` option, this option can be used to explicitly specify the kernel type to be CUDA.

OpenCL-specific options
-----------------------

``--local_size=...``
^^^^^^^^^^^^^^^^^^^^

Specify whether work-group is 1D, 2D 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D work group, respectively. This corresponds to the `local_work_size` parameter of clEnqueueNDRangeKernel().

``--num_groups=...``
^^^^^^^^^^^^^^^^^^^^

Specify whether grid of work-groups is 1D, 2D or 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D grid, respectively. This argument and ``--global_size=`` are mutually exclusive.

``--global_size=...``
^^^^^^^^^^^^^^^^^^^^^

Specify whether NDRange is 1D, 2D 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D NDRange, respectively. This corresponds to the `global_work_size` parameter of clEnqueueNDRangeKernel(). This argument and ``--num-groups=`` are mutually exclusive.

CUDA-specific options
---------------------

``--blockDim=...``
^^^^^^^^^^^^^^^^^^

Specify whether thread block is 1D, 2D or 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D thread block, respectively.

``--gridDim=...``
^^^^^^^^^^^^^^^^^
Specify whether grid of thread blocks is 1D, 2D or 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D grid, respectively.

Advanced options
----------------

``--pointer-bitwidth=``\{32, 64}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the bitwidth of the pointers used in the kernel. This can either be 32-bit or 64-bit. The default is 32-bit.

Currently some invariant inference rules are disabled in when 64-bit pointers are used. Hence, not all kernels that will verify with 32-bit pointers also verify with 64-bit pointers.

.. _adversarial-abstraction:

``--adversarial-abstraction``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Completely abstract shared state, so that reads are nondeterministic.

.. todo:: Give small example illustrating how drastic this can be.

.. todo:: Justify why it can be useful (performance)

See also :ref:`equality-abstraction`.

.. _equality-abstraction:

``--equality-abstraction``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Make shared arrays nondeterministic, but consistent between work items, at barriers.

.. todo: Give example of what this lets you do and where it is not enough.

See also :ref:`adversarial-abstraction`.

``--asymmetric-asserts``
^^^^^^^^^^^^^^^^^^^^^^^^

When "dualizing" an assertion, generate the assertion only for the first thread under consideration.  This is sound, because the thread is arbitrary, and can lead to faster verification, but can also yield false positives.

.. todo: I [Ally] do not understand why this could lead to false positives.  Is it because a loop invariant only gets assumed for one of the threads?  Would it be OK to turn assert(phi) into: assert(phi$1); assume(phi$2)?  This might be sound and not suffer from the false positive issue.

``--boogie-file=``\X\ ``.bpl``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify a supporting ``.bpl`` file to be used during verification.  This file is passed, unmodified, to Boogie when verification is performed.  This can be useful, for example, if you wish to declare an uninterpreted function and use it in your kernel, and then provide some axioms about the function for Boogie to use during reasoning.

``--math-int``
^^^^^^^^^^^^^^

Represent integer types using mathematical integers instead of bit-vectors.

``--no-annotations``
^^^^^^^^^^^^^^^^^^^^

Ignore all source-level annotations.

``--only-requires``
^^^^^^^^^^^^^^^^^^^

Ignore all source-level annotations except for requires.

``--invariants-as-candidates``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interpret all source-level invariants as candidate invariants. This means they
will be automatically removed by the Houdini procedure in case they do not hold. No error will be issued.

``--no-barrier-access-checks``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off access checks for barrier invariants.

``--no-inline``
^^^^^^^^^^^^^^^

Turn off automatic function inlining and use modular verification instead.

In some cases this may speed-up verification. However, it is more likely that verification will fail, as GPUVerify's procedure contract inference capabilities are rather limited.

``--only-log``
^^^^^^^^^^^^^^

Log accesses to arrays, but do not check for races. This can be useful for determining access pattern invariants.

``--kernel-args=``\K,v\ :sub:`1`\ ,...,v\ :sub:`n`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For kernel K with scalar parameters ``x``\ :sub:`1`\ , ..., ``x``\ :sub:`n`\ , add the preconditions ``x``\ :sub:`1`  == v\ :sub:`1`\ , ..., ``x``\ :sub:`1`  == v\ :sub:`n`\ . Use ``*`` to denote an unconstrained parameter.

``--kernel-arrays=``\K,s\ :sub:`1`\ ,...,s\ :sub:`n`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For kernel K with array parameters (``p``\ :sub:`1`\ , ..., ``p``\ :sub:`n`\ ), assume that sizeof(``p``\ :sub:`1`\ ) == s\ :sub:`1`\ , ... sizeof(``p``\ :sub:`n`\ ) == s\ :sub:`n`\ . Use ``*`` to denote an unconstrained size.

``--warp-sync=``\X
^^^^^^^^^^^^^^^^^^

Synchronize threads within warps, sized X.

``--race-instrumenter=``\{original,watchdog-single,watchdog-multiple}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose which method of race instrumentation to use. The default is watchdog-single.

``--solver=``\{z3,cvc4}
^^^^^^^^^^^^^^^^^^^^^^^

Choose which SMT solver to use in the back-end. The default depends on the settings in the ``gvfindtools.py`` file.

``--stop-at-opt``
^^^^^^^^^^^^^^^^^

Stop after the LLVM optimization pass.

``--stop-at-gbpl``
^^^^^^^^^^^^^^^^^^

Stop after generating a gbpl file.

.. _stop-at-bpl:

``--stop-at-bpl``
^^^^^^^^^^^^^^^^^

Stop after generating a bpl file.


``--stop-at-cbpl``
^^^^^^^^^^^^^^^^^^

Stop after generating an annotated bpl

Development Options
-------------------

``--debug``
^^^^^^^^^^^

Normally, GPUVerify suppresses exceptions, dumping them to a file and printing a standard "internal error" message.  This option turns off this suppression.

.. _keep-temps:

``--keep-temps``
^^^^^^^^^^^^^^^^

Keep intermediate bc, gbpl, bpl and cbpl files.

``--gen-smt2``
^^^^^^^^^^^^^^

Generate an smt2 file. The file contains the queries sent to the SMT solver.

``--clang-opt=...``
^^^^^^^^^^^^^^^^^^^

Use this option to pass a command-line option directly to Clang, the front-end used by GPUVerify.

``--opt-opt=...``
^^^^^^^^^^^^^^^^^^^

Use this option to pass a command-line option directly to opt, the LLVM optimizer used by GPUVerify.

``--bugle-opt=...``
^^^^^^^^^^^^^^^^^^^

Use this to pass a command-line option directly to Bugle, the component of GPUVerify that translates LLVM bitcode into Boogie.

``--vcgen-opt=...``
^^^^^^^^^^^^^^^^^^^

Specify a command-line option to be passed to the VC generator.

``--cruncher-opt=``...
^^^^^^^^^^^^^^^^^^^^^^

Specify an option to be passed directly to the cruncher.

.. _boogie-opt:

``--boogie-opt=``...
^^^^^^^^^^^^^^^^^^^^

Specify an option to be passed directly to Boogie.  For instance, if you want to see what Boogie is doing, you can use ``--boogie-opt=/trace``.  In this case you also need to pass :ref:`verbose` to GPUVerify.

Invariant inference options
---------------------------

``--no-infer``
^^^^^^^^^^^^^^

Turn off invariant inference.

``--omit-infer=``\X
^^^^^^^^^^^^^^^^^^^

Do not generate invariants of type 'X'.

``--infer-info``
^^^^^^^^^^^^^^^^

Prints information about the inference process.

``--k-induction-depth``\X
^^^^^^^^^^^^^^^^^^^^^^^^^

Applies k-induction with k=X to all loops.
