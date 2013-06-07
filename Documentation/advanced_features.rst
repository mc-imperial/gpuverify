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
larger than 128 the invariant is not maintained by the loop, and
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
