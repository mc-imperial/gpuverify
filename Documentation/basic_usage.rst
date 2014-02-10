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
``--global_size=`` and ``--local_size`` arguments have a direct correspondance
to the ``global_work_size`` and ``local_work_size`` parameters to the OpenCL
Runtime function ``clEnqueueNDRangeKernel()`` respectively.

Here is an example using ``--global_size=``::

  gpuverify --local_size=[32,32] --global_size=[512,512] kernel.cl

Which is equivilant to the following using ``--num_groups=`` instead of ``--global_size=``::

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

In the description of command line options, we follow OpenCL terminology, not CUDA terminology.  We thus refer to work items and work groups, not threads and thread blocks, and to local memory, not shared memory.

General options
---------------

-h, --help
^^^^^^^^^^

Display list of GPUVerify options.  Please report cases where
GPUVerify claims to have an option not documented here, or if an
option mentioned here is not listed by GPUVerify.

-I <value>
^^^^^^^^^^

Add directory to include search path

-D <value>
^^^^^^^^^^

Define symbol

``--findbugs``
^^^^^^^^^^^^^^

Run tool in bug-finding mode, see :ref:`findbugs`.  In this mode, loop invariant inference
is disabled, and a loop unwinding depth of 2 is used, unless this
depth is over-ridden using ``--loop-unwind``.

``--loop-unwind=``\X
^^^^^^^^^^^^^^^^^^^^

Run tool in *findbugs* mode (see :ref:`findbugs`) and explore only traces that pass through at most X loop heads.

``--memout=``\X
^^^^^^^^^^^^^^^

Give Boogie, the verifier on which GPUVerify is built, a hard memory
limit of X megabytes.  Specifying a memout of 0 disables the
memout. The default is 0, i.e. no memory limit.

``--no-benign``
^^^^^^^^^^^^^^^

By default, GPUVerify tries to tolerate certain kinds of (arguably)
*benign* data races.  For example, if GPUVerify can figure out that in
a write-write data race, both work items involved are guaranteed to write
the same value to the memory location in question, it will not report
the race.

Sometimes we wish to turn off this tolerance, perhaps because we
believe our kernel should be free from such races, or because we are
feeling strict and want to take the (arguably correct) view that "all
data races are evil with no exceptions".

.. todo:: Add link to the paper with this title.

Also, it may be the case (though we have not evaluated this
systematically) that tolerating benign races carries some performance
overhead in terms of verification time.

To disable tolerance of benign races, specify ``--no-benign``.

.. todo:: Maybe this option should be ``--no-benign-tolerance``.  Just ``--no-benign`` is perhaps a bit misleading: one might think it means "don't warn be about benign races"; actually, it means the opposite.

``--only-divergence``
^^^^^^^^^^^^^^^^^^^^^

Disable race checking, and only check for barrier divergence.

``--only-intra-group``
^^^^^^^^^^^^^^^^^^^^^^

Do not check for inter-work-group races.  In this mode, a kernel may be deemed correct even if it can exhibit races on global memory between work items in different work groups, as long as GPUVerify can prove that there are no data races (on global or local memory) between work items in the same work group.

``--time``
^^^^^^^^^^

When GPUVerify finishes, print some statistics about how long it took.

``--timeout=``\X
^^^^^^^^^^^^^^^^

Allow Boogie to run for X seconds before giving up.  Specifying 0 disables the timeout. The default is 300 seconds.

``--verify``
^^^^^^^^^^^^

Run GPUVerify in *verify* mode (see :ref:`verifymode`).  This is the mode the tool uses by default.

.. todo:: Perhaps this option should go?

.. _verbose:

``--verbose``
^^^^^^^^^^^^^

With this option, GPUVerify will print the various sub-commands that are issued during the analysis process.  Also, output produced by the tools which GPUVerify invokes will be displayed.  If you are debugging, and are issuing print statements in one of the GPUVerify components, you will need to use ``--verbose`` to be able to see the results of this printing.

``--version``
^^^^^^^^^^^^^

Show version information.


Advanced options
----------------

.. _adversarial-abstraction:

``--adversarial-abstraction``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Completely abstract shared state, so that reads are nondeterministic.

.. todo:: Give small example illustrating how drastic this can be.

.. todo:: Justify why it can be useful (performance)

See also :ref:`equality-abstraction`.

``--array-equalities``
^^^^^^^^^^^^^^^^^^^^^^

During invariant inference, generate equality candidate invariants for array variables.  This is not done by default as it can be very expensive.

``--asymmetric-asserts``
^^^^^^^^^^^^^^^^^^^^^^^^

When "dualising" an assertion, generate the assertion only for the first thread under consideration.  This is sound, because the thread is arbitrary, and can lead to faster verification, but can also yield false positives.

.. todo: I [Ally] do not understand why this could lead to false positives.  Is it because a loop invariant only gets assumed for one of the threads?  Would it be OK to turn assert(phi) into: assert(phi$1); assume(phi$2)?  This might be sound and not suffer from the false positive issue.

``--boogie-file=``\X\ ``.bpl``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify a supporting ``.bpl`` file to be used during verification.  This file is passed, unmodified, to Boogie when verification is performed.  This can be useful, for example, if you wish to declare an uninterpreted function and use it in your kernel, and then provide some axioms about the function for Boogie to use during reasoning.

.. _boogie-opt:

``--boogie-opt=``...
^^^^^^^^^^^^^^^^^^^^

Specify an option to be passed directly to Boogie.  For instance, if you want to see what Boogie is doing, you can use ``--boogie-opt=/trace``.  (In this case you also need to pass :ref:`verbose` to GPUVerify.)

``--bugle-lang=[cl|cu]``
^^^^^^^^^^^^^^^^^^^^^^^^

If you run GPUVerify directly on an LLVM bitcode file, you'll need to tell Bugle whether the bitcode originated from an OpenCL or CUDA kernel.  This option lets you do so.

``--bugle-opt=...``
^^^^^^^^^^^^^^^^^^^

Use this to pass a command-line option directly to Bugle, the component of GPUVerify that translates LLVM bitcode into Boogie.

``--call-site-analysis``
^^^^^^^^^^^^^^^^^^^^^^^^

Turn on call site analysis.

.. todo: I [Ally] do not know what this analysis is.

``--clang-opt=...``
^^^^^^^^^^^^^^^^^^^

Use this option to pass a command-line option directly to Clang, the front-end used by GPUVerify.

``--debug``
^^^^^^^^^^^

In "customer-facing" mode, GPUVerify suppresses exceptions, dumping them to a file and printing a standard "internal error" message.  This option turns off this suppression, to make it faster to debug GPUVerify.

.. _equality-abstraction:

``--equality-abstraction``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Make shared arrays nondeterministic, but consistent between work items, at barriers.

.. todo: Give example of what this lets you do and where it is not enough.

See also :ref:`adversarial-abstraction`.

``--gen-smt2``
^^^^^^^^^^^^^^

.. todo: From here onwards I have pretty much just pasted from the -h option of GPUVerify.  Some of the options will need more explanation.

Generate smt2 file

.. _keep-temps:

``--keep-temps``
^^^^^^^^^^^^^^^^

Keep intermediate bc, gbpl, bpl and cbpl files

``--math-int``
^^^^^^^^^^^^^^

Represent integer types using mathematical integers instead of bit-vectors

``--no-annotations``
^^^^^^^^^^^^^^^^^^^^

Ignore all source-level annotations

``--only-requires``
^^^^^^^^^^^^^^^^^^^

Ignore all source-level annotations except for requires

``--no-barrier-access-checks``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off access checks for barrier invariants

``--no-constant-write-checks``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off access checks for writes to constant space

``--no-inline``
^^^^^^^^^^^^^^^

Turn off automatic inlining by Bugle

``--no-loop-predicate-invariants``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off automatic generation of loop invariants related to predicates, which can be incorrect

``--no-smart-predication``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off smart predication

``--no-source-loc-infer``
^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off inference of source location information

``--no-uniformity-analysis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off uniformity analysis

``--only-log``
^^^^^^^^^^^^^^

Log accesses to arrays, but do not check for races. This can be useful for determining access pattern invariants

``--silent``
^^^^^^^^^^^^

Silent on success; only show errors/timing

``--stop-at-opt``
^^^^^^^^^^^^^^^^^

Stop after LLVM optimization pass

``--stop-at-gbpl``
^^^^^^^^^^^^^^^^^^

Stop after generating gbpl

``--stop-at-cbpl``
^^^^^^^^^^^^^^^^^^

Stop after generating an annotated bpl

.. _stop-at-bpl:

``--stop-at-bpl``
^^^^^^^^^^^^^^^^^

Stop after generating bpl

``--time-as-csv=``\label
^^^^^^^^^^^^^^^^^^^^^^^^

Print timing as CSV row with label

``--vcgen-timeout=``\X
^^^^^^^^^^^^^^^^^^^^^^

Allow VCGen to run for X seconds.

``--vcgen-opt=...``
^^^^^^^^^^^^^^^^^^^

Specify option to be passed to be passed to VC generation engine

``--warp-sync=``\X
^^^^^^^^^^^^^^^^^^

Synchronize threads within warps, sized X, defaulting to 32

.. todo: Sounds like this is on by default, but it is not.  So what does "default" mean here?

``--atomic=``\X
^^^^^^^^^^^^^^^

Check atomics as racy against reads (r), writes (w), both (rw), or none (none) (default is ``--atomic=rw``)

.. todo: Should this go, now that OpenCL 2 suggests what the rules are?

``--no-refined-atomics``
^^^^^^^^^^^^^^^^^^^^^^^^
Don't do abstraction refinement on the return values from atomics

``--solver=``\X
^^^^^^^^^^^^^^^

Choose which SMT Theorem Prover to use in the backend.  Available options: 'Z3' or 'cvc4' (default is 'Z3')

``--logic=X``
^^^^^^^^^^^^^

Define the logic to be used by the CVC4 SMT solver backend (default is QF_ALL_SUPPORTED)

Invariant inference options
---------------------------

``--no-infer``
^^^^^^^^^^^^^^

Turn off invariant inference

``--infer-timeout=``\X
^^^^^^^^^^^^^^^^^^^^^^

Allow GPUVerifyCruncher to run for X seconds.

``--staged-inference``
^^^^^^^^^^^^^^^^^^^^^^

Perform invariant inference in stages; this can sometimes boost performance for complex kernels

``--parallel-inference``
^^^^^^^^^^^^^^^^^^^^^^^^

Use multiple solver instances in parallel to potentially accelerate invariant inference

``--dynamic-analysis``
^^^^^^^^^^^^^^^^^^^^^^

Use dynamic analysis to falsify invariants.

``--scheduling=``\X
^^^^^^^^^^^^^^^^^^^

Choose a parallel scheduling strategy from the following: 'default', 'unsound-first' or 'brute-force'. The 'default' strategy executes first any dynamic engines, then any unsound static engines and then the sound static engines. The 'unsound-first' strategy executes any unsound engines (either static or dynamic) together before the soundengines.  The 'brute-force' strategy executes all engines together but performance is highly non-deterministic.

``--infer-config-file=``\X\ ``.cfg``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify a custom configuration file to be used during invariant inference

``--infer-info``
^^^^^^^^^^^^^^^^

Prints information about the inference process.

OpenCL-specific options
-----------------------

``--local_size=...``
^^^^^^^^^^^^^^^^^^^^

Specify whether work-group is 1D, 2D 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D work group, respectively.
This corresponds to the `local_work_size` parameter of clEnqueueNDRangeKernel().  

``--num_groups=...``
^^^^^^^^^^^^^^^^^^^^

Specify whether grid of work-groups is 1D, 2D or 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a !D, 2D and 3D grid, respectively.
This argument and ``--global_size=`` are mutually exclusive.

``--global_size=...``
^^^^^^^^^^^^^^^^^^^^^

Specify whether NDRange is 1D, 2D 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D NDRange, respectively.
This corresponds to the `global_work_size` parameter of clEnqueueNDRangeKernel(). This argument and ``--num-groups=`` are mutually exclusive.

CUDA-specific options
---------------------

``--blockDim=...``
^^^^^^^^^^^^^^^^^^

Specify whether thread block is 1D, 2D or 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a 1D, 2D and 3D thread block, respectively.

``--gridDim=...``
^^^^^^^^^^^^^^^^^
Specify whether grid of thread blocks is 1D, 2D or 3D and specify size for each dimension.  Use X, [X,Y] and [X,Y,Z] for a !D, 2D and 3D grid, respectively.


