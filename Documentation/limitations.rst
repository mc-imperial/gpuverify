===========
Limitations
===========

Known Sources of Unsoundness in GPUVerify
-----------------------------------------

* By defualt, GPUVerify does not perform any out-of-bounds checking. Hence, a
  verified kernel might still behave unexpectedly when array bounds are
  violated.

* GPUVerify assumes the input arrays to a kernel are all distinct. Not making
  this assumption leads to many false positives. Non-aliasing can be enforced
  by equipping the input arrays with restrict qualifiers. GPUVerify will issue
  a warning when the restrict qualifier is missing and there is more than one
  input array. For example, for the the following CUDA kernel::

    __global__ void foo(int* a, int* b) {
      a[threadIdx.x] = b[threadIdx.x];
    }

  GPUVerify will report it is assuming the the arguments ``a`` and ``b`` of
  ``foo`` to be non-aliased. This warning can be suppressed by changing the
  kernel to::

    __global__ void foo(int* __restrict__ a, int* __restrict__ b) {
      a[threadIdx.x] = b[threadIdx.x];
    }

* GPUVerify's default pointer representation may cause false negatives to occur
  in kernel verification if the kernel explicitly tries to enumerate all
  possible pointer values (effectively: all potential input arrays plus
  ``NULL``). For example, the following CUDA kernel will successfully verify::

    __device__ int *nondet(void);

    __global__ void foo(int* a, int* b) {
      int* p = nondet();
      int* q = nondet();

      __assume(__ptr_offset_bytes(p) == 0);
      __assume(__ptr_offset_bytes(q) == 0);

      if (p != a && p != b && p != NULL &&
          q != a && q != b && q != NULL && p != q) {
        __assert(false);
      }

    }

* GPUVerify assumes that atomic operations do not overflow or underflow the
  values on which they act.

* If an array ``a`` occurs read-only in a kernel, then the ``__read(a)``
  annotation always evaluates to false. As a consequence, the following OpenCL
  kernel successfully verifies::

    __kernel void foo(__global int* a, __global int* b) {
      b[get_global_id(0)] = a[0];
      __assert(!__read(a));
    }

Using Structures as Kernel Parameters
-------------------------------------

By default the clang compiler as used by GPUVerify replace any structure passed
to a kernel by a char pointer plus a memory copy. For example, the following
kernel::

  struct S {
    int *a;
    int *b;
  };

  __global__ void foo(struct S s) {
  }

will internally represented by::

  __global__ void foo(char *s_in) {
    struct S s;
    memcpy(&s, s_in, sizeof(s));
  }

Consequently, although it is possible to write kernel preconditions like
``__requires(s.a == NULL)``, these will be ignored during the verification.
As a workaround ``__assume(s.a == NULL)`` can be used.
