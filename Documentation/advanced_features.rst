===============================
Advanced Features
===============================

Annotation language
===================

Expert users can make use of a variety of builtin functions to
annotate their code to aid verification.  We now describe these annotations.

Assertions
----------

You can write assertions in GPU kernels using the special function
``__assert``.  Such assertions will be checked by GPUVerify.  For
example, consider the following CUDA kernel: 

.. literalinclude:: examples/failing-assert.cu

If we run GPUVerify specifying ``blockDim=8`` and ``gridDim=8`` then the tool verifies that the assertion cannot fail.  However, for large block and grid sizes it can fail::

  gpuverify --blockDim=64 --gridDim=64 failing-assert.cu

.. literalinclude:: examples/failing-assert-error.txt

Loop invariants
---------------

GPUVerify may be unable to verify a kernel due to limitations in
the tool's invariant inference engine.  In this case the user can
supply invariants manually.  However, any manually supplied invariant
is checked by the tool.

If an assertion is placed *exactly* at the head of a loop, it
is treated by GPUVerify as a user-supplied loop invariant.  For
example, in the following (dumb) OpenCL kernel:

.. literalinclude:: examples/assert-as-invariant.cl

the assertion ``__assert(i <= 100)`` is taken to be a loop
invariant because it appears directly at the head of the loop.  The
comma operator is used to allow invariant assertions to precede the
loop guard.  In fact, invariant assertions are actually supplied as
*part of* the loop guard expression, but they do not affect the
truth value of the loop guard.

If the above example is analysed with a sufficiently small local
size, the invariant is verified as holding.  However, for a local size
larger than 101 the invariant is not maintained by the loop, and
GPUVerify detects this potential problem::

  gpuverify --local_size=128 --num_groups=16 assert-as-invariant.cl

.. literalinclude:: examples/assert-as-invariant-error.txt

For readability, one can write ``__invariant`` as a synonym for ``__assert``.  This is recommended to state the intention that a given assertion is an invariant rather than a plain assertion, but currently GPUVerify does not check that this intention is satisifed.  In particular, if ``__invariant`` is used somewhere other than a loop head then it is treated as a plain assertion.

Multiple invariants for a loop can be specified using multiple ``__invariant`` or ``__assert`` commands at the loop head.  The following example illustrates this, and also shows how an invariant can be specified for a ``for`` loop:

.. literalinclude:: examples/invariant-for-loop.cl

Notice that three invariants have been specified, two using
``__invariant`` and one using ``__assert``.  Notice also
that with a ``for`` loop, the invariants come immediately before
the guard, which is the middle component of the ``for`` loop's
specifier.  Again, the comma operator is used to separate the
invariants from the guard.

For sufficiently small local sizes these invariants hold.  However,
for a large local size, the first invariant will fail on loop entry,
as reported by GPUVerify::

  gpuverify --local_size=1024 --num_groups=16 invariant-for-loop.cl

.. literalinclude:: examples/invariant-for-loop-error.txt

Invariants are easy to specify for ``do-while`` loops: the invariants must simply appear as the first commands of the loop body:

.. literalinclude:: examples/invariant-do-while-loop.cl

While programming using ``goto`` is not recommended, the ``goto`` keyword may be used in OpenCL.  If you write a loop using ``goto`` and wish to specify invariants for this loop then the rule is the same as usual: place invariants at the loop head.  This is illustrated for a ``goto`` version of the ``do-while`` example as follows:

.. literalinclude:: examples/invariant-goto-loop.cl

Pre-conditions
--------------

Sometimes a kernel is correct only for certain input parameter values.  For example, the following CUDA kernel is not correct when executed by a single block of 1024 threads for *arbitrary* values of parameter ``sz``:

.. literalinclude:: examples/needs-requires.cu

There will be data races if the absolute value of ``sz`` is less than
the size of a thread block, or if the absolute value of ``sz`` is so
large that overflow can occur.  GPUVerify therefore rejects the
kernel:

.. literalinclude:: examples/needs-requires-error.txt

If the programmer knows that ``sz`` should be equal to the
``x`` dimension of a thread block, they can specify this via a
*pre-condition* on the kernel, using he ``__requires``
annotation:

.. literalinclude:: examples/with-requires.cu

With this annotation, GPUVerify is able to verify race-freedom.

**Rules for using ``__requires``**:
.. todo:: Be careful with short-circuit evaluation

 
Post-conditions
---------------

.. todo:: Need to describe ``__return_val`` and ``__old``


Assume statements
-----------------

.. todo:: Say that this should be used with extreme care


Global assertions and pre-/post-conditions
------------------------------------------

.. todo:: This will explain:

  * ``__global_requires``

  * ``__global_ensures``

  * ``__global_assert``


Reasoning about memory accesses
-------------------------------

.. todo:: ``__write_implies``, etc.  Be sure to comment on byte-level reasoning 
          issue.


Looking into automatic invariant generation
===========================================

.. _inspecting-invariants:

Inspecting invariants generated by GPUVerify
--------------------------------------------

If you are looking to extend GPUVerify's invariant inference capabilities, it can be useful to look at the invariants the tool is generating.

To do this, run GPUVerify with the :ref:`keep-temps` option.  If you just want to see the candidate invariants the tool generates, and do not wish for verification to be attempted, you can use :ref:`stop-at-bpl` to tell GPUVerify to stop once is has generated the Boogie program that contains candidate invariants.

Assuming that your kernel was called ``kernel.cl`` or ``kernel.cu``, the :ref:`keep-temps` option will lead to a file called ``kernel.bpl`` being created.

If you look in this file and search for ``:tag``, you may find some assertions that have the ``:tag`` attribute.  This attribute is a string indicating which invariant generation rule inside the ``GPUVerifyVCGen`` component of GPUVerify led to the candidate invariant being generated.

For example, if you try this on::

  testsuite/OpenCL/test_mod_invariants/global_direct/kernel.cl

then in the ``.bpl`` file you will see something like::

  $1:
    assert {:tag "accessedOffsetsSatisfyPredicates"} _b2 ==> _WRITE_HAS_OCCURRED_$$A ==> BV32_AND(BV32_SUB(256bv32, 1bv32), _WRITE_OFFSET_$$A) == BV32_AND(BV32_SUB(256bv32, 1bv32), BV32_ADD(BV32_MUL(group_size_x, group_id_x$1), local_id_x$1));
    assert {:tag "guardNonNeg"} {:thread 1} p0$1 ==> _b1 ==> BV32_SLE(0bv32, $i.0$1);
    assert {:tag "guardNonNeg"} {:thread 2} p0$2 ==> _b1 ==> BV32_SLE(0bv32, $i.0$2);
    assert {:tag "loopCounterIsStrided"} {:thread 1} p0$1 ==> _b0 ==> BV32_AND(BV32_SUB(256bv32, 1bv32), $i.0$1) == BV32_AND(BV32_SUB(256bv32, 1bv32), BV32_ADD(BV32_MUL(group_size_x, group_id_x$1), local_id_x$1));
    assert {:tag "loopCounterIsStrided"} {:thread 2} p0$2 ==> _b0 ==> BV32_AND(BV32_SUB(256bv32, 1bv32), $i.0$2) == BV32_AND(BV32_SUB(256bv32, 1bv32), BV32_ADD(BV32_MUL(group_size_x, group_id_x$2), local_id_x$2));
    assert {:tag "user"} {:originated_from_invariant} {:line 12} {:col 7} {:fname "kernel.cl"} {:dir "/Users/nafe/work/autobuild/mac/gpuverify/testsuite/OpenCL/test_mod_invariants/global_direct"} {:thread 1} p0$1 ==> _c0 ==> (if $i.0$1 == v0$1 then 1bv1 else 0bv1) != 0bv1;
    assert {:tag "user"} {:originated_from_invariant} {:line 12} {:col 7} {:fname "kernel.cl"} {:dir "/Users/nafe/work/autobuild/mac/gpuverify/testsuite/OpenCL/test_mod_invariants/global_direct"} {:thread 2} p0$2 ==> _c0 ==> (if $i.0$2 == v0$2 then 1bv1 else 0bv1) != 0bv1;

(Of course, the exact form and number of invariants generated may change, as it is sensitive to the test case in question and the current state of the candidate generation engine.)

The candidate invariants can be pretty difficult to read!  They have not been designed for human consumption.

Watching GPUVerify eliminate invariants
---------------------------------------

Each invariant candidate has the form ``_bX ==> e``, where ``_bX`` is a Boolean constant marked with the ``:existential`` attribute.  This tells Boogie that ``_bX`` should be treated specially: it should be used by the Houdini algorithm to turn the given invariant on or off.  Clearly if ``_bX`` is set to ``true``, it is as if the invariant ``e`` appeared directly, while if ``_bX`` is set to ``false`` it is as if no invariant were present at all (because ``false ==> e`` is equivalent to the trivial invariant ``true``).

Houdini starts by setting all of the ``_b`` variables to ``true`` and trying to prove that they form an inductive invariant for the program.  If they do not, Houdini will kick out at least one candidate, by setting its ``_b`` variable to ``false``, and the process continues.

To see this kicking out of candidates in action, you can run GPUVerify with :ref:`verbose` and with ``--boogie-opt=/trace`` (see :ref:`boogie-opt`).  If you try this on::

  testsuite/OpenCL/test_mod_invariants/global_direct/kernel.cl

You'll see something like this in the output::

    Verifying $foo
    Houdini assignment axiom: (And true _b0 _b1 _b2 _b3 _b4 _b5 _b6 _b7 _b8)
    Time taken = 0.156258
    Removing _b6
    Removing _b2
    Verifying $foo
    Houdini assignment axiom: (And true _b0 _b1 (! _b2) _b3 _b4 _b5 (! _b6) _b7 _b8)

    Time taken = 0.0781293

The first Houdini assignment axiom sets all of the ``_b`` variables to ``true``.  Boogie then tries to verify the kernel, and it finds that ``_b6`` and ``_b2`` (i.e. the ``access lower bound`` candidates shown in :ref:`inspecting-invariants`) do not hold.  In the next Houdini iteration, these candidates are flipped to ``false``.  The remaining invariants are then found to hold.  With this invariant, the tool then proceeds to do race analysis (not shown in the above snippet).

Barrier invariants
==================

.. todo:: Simple example of barrier invariant usage

Uninterpreted functions
=======================

.. todo:: Show how UFs can be used to model operations abstractly

Procedure specifications
========================

.. todo:: Show (currently hacky) way of providing a procedure spec
          without body

Passing arguments to Boogie
===========================

.. todo:: Give example use of ``--boogie-opt``

Linking with Boogie files
=========================

.. todo:: Describe how an external ``.bpl`` file can be used.
