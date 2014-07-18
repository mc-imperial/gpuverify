A very brief introduction to GPUVerify
======================================

The aim of this tutorial is to help new users get up and running with
GPUVerify quickly.

.. contents::

For more detailed instructions on using GPUVerify take a look at:

- GPUVerify documentation:
  http://multicore.doc.ic.ac.uk/tools/GPUVerify/docs/index.html

- Introductory YouTube video:
  https://www.youtube.com/watch?v=l8ysBPV8OvA


Prerequisites
-------------

* Python 2.7 or higher.  GPUVerify is coordinated by a Python script,
  so you must have Python 2.7 or a more recent version installed and
  on your path.

  Windows users can obtain a Python binary here:
  https://www.python.org/downloads/windows/

  Python comes as standard with most Linux distributions and can be
  obtained and updated using your package manager.

  Python comes as standard with Mac OS X. However, we recommend installing
  Python using Homebrew.

* psutil.  GPUVerify relies on the psutil Python module.

  Windows users can obtain a binary version of this module here:
  https://pypi.python.org/pypi?:action=display&name=psutil#downloads

  Linux/Mac OS X users can obtain psutil using pip as follows:

  ::

    pip install psutil

* mono (Linux/Max OS X only).  Linux/Mac OS X users must have a recent version
  of mono installed.  Please see the GPUVerify documentation for details of how
  to obtain this:
  http://multicore.doc.ic.ac.uk/tools/GPUVerify/docs/developer_guide.html


Running GPUVerify on some kernels
---------------------------------


**Important note for Windows users**

Before you extract the GPUVerify archive you need to "unblock" it.

- Right click on the .zip file
- Click "Properties"
- At the bottom of the "General" tab you should see:

  ::

     Security: This file came from another computer and might be
     blocked to help protect this computer.

- Click "Unblock"
- Click "OK"


**Adding GPUVerify to your path**

After extracting the GPUVerify download, you should add the directory
into which you have extracted GPUVerify to your ``PATH`` environment
variable.  This directory contains the ``GPUVerify.py`` script.

**Finding a race in a simple kernel**

Open a Windows command prompt or Linux/OSX terminal, and navigate to:

::

  testsuite/OpenCL/misc/fail/2d_array_race

This is one of several hundred kernels that comprise the GPUVerify
test suite.

Have a look at ``kernel.cl``.  This is a small kernel that manipulates a
2D array in a racy manner.

The top three comment lines are there for GPUVerify's testing tool and
you can ignore them.

Let's try to verify that this kernel is free from data races when
executed by a whole load of threads: a 2D grid of work groups, of
dimensions 256x256, where each work group is also 2D with dimensions
64*64.

To do this, enter the command:

::

  GPUVerify --local_size=64,64 --num_groups=256,256 kernel.cl

After a few seconds GPUVerify reports some data races for this kernel.
For instance, you'll see something like this:

::

  kernel.cl: error: possible write-read race on L[32][0]:

  Read by work item (0, 31) in work group (138, 0), kernel.cl:11:7:
    L[get_local_id(1)+1][get_local_id(0)]++;

  Write by work item (0, 32) in work group (138, 0), kernel.cl:9:7:
    L[get_local_id(1)][get_local_id(0)] = G[get_global_id(1)*get_global_size(1) +
  get_global_id(0)];

GPUVerify has identified that two work items in work group (138, 0)
can race; in particular the work items with local ids (0, 31) and (0,
32) inside that group.


**Verifying a simple kernel**

Navigate to:

::

  testsuite/OpenCL/async_work_group_copy/pass/test1

and take a look at kernel.cl.  This example make use of OpenCL
asynchronous memory copying, and it uses this feature in a race-free
manner.

Let's try to verify that this kernel is free from data races when
executed by 128 work groups, each comprised of 64 work items:

::

  GPUVerify --local_size=64 --num_groups=128 kernel.cl

After a few seconds you should find that verification was successful:

::

  GPUVerify kernel analyser finished with 1 verified, 0 errors
  Verified: kernel.cl
  - no data races within work groups
  - no data races between work groups
  - no barrier divergence
  - no assertion failures
  (but absolutely no warranty provided)

Troubleshooting common problems
-------------------------------


**Do you definitely have Python?**

If you see something like:

"Requested Python version () is not installed"
or
"/usr/bin/env: python: No such file or directory"

then you need to get Python.  See "Prerequisites" above.


**Do you definitely have psutil?**

If you get a message like:

"GPUVerify requires Python to be equipped with the psutil module."

then you need to get the psutil Python module.  See "Prerequisites"
above.


**Did you unblock the GPUVerify .zip file before extracting it?**

If under Windows you get a message whose contents includes:

"An attempt was made to load an assembly from a network location"

then you forgot to unblock the GPUVerify .zip file before you
extracted it.  Please follow the instructions at the start of Section
1 above for info on how to do this.


**GPUVerify is taking a long time to give an answer for my kernel**

Try running GPUVerify with the --infer-info option.  With this option
the tool should periodically report on its progress.  You'll see output
like this (the following is specific to a particular kernel; the
output you'll see for your kernel will have the same form but will
differ a lot in the details):

::

  Verifying $binomial_options_kernel
  Houdini assignment axiom: (And true _b0 _b1 _b2 _b3 _b4 _b5 _b6 _b7 _b8 _b9 _b10
   _b11 _b12 _b13 _b14 _b15 _b16 _b17 _b18 _b19 _b20 _b21 _b22 _b23 _b24 _b25 _b26
   _b27 _b28 _b29 _b30 _b31 _b32 _b33 _b34 _b35 _b36 _b37 _b38 _b39 _b40 _b41 _b42
   _b43 _b44 _b45 _b46 _b47 _b48 _b49 _b50 _b51 _b52 _b53 _b54 _b55 _b56 _b57 _b58
   _b59 _b60 _b61 _b62 _b63 _b64 _b65 _b66 _b67 _b68)

Then after some time you'll see something like:

::

  Time taken = 2.7760232
  Removing _b4
  Removing _b28
  Removing _b4
  Removing _b22
  Removing _b63
  Removing _b63
  Removing _b65

Then you'll see another verification step:

::

  Verifying $binomial_options_kernel
  Houdini assignment axiom: (And true _b0 _b1 _b2 _b3 (! _b4) _b5 _b6 _b7 _b8 _b9
  _b10 _b11 _b12 _b13 _b14 _b15 _b16 _b17 _b18 _b19 _b20 _b21 (! _b22) _b23 _b24 _
  b25 _b26 _b27 (! _b28) _b29 _b30 _b31 _b32 _b33 _b34 _b35 _b36 _b37 _b38 _b39 _b
  40 _b41 _b42 _b43 _b44 _b45 _b46 _b47 _b48 _b49 _b50 _b51 _b52 _b53 _b54 _b55 _b
  56 _b57 _b58 _b59 _b60 _b61 _b62 (! _b63) _b64 (! _b65) _b66 _b67 _b68)

This process may continue for some time.

If you observe this behaviour then GPUVerify is slowly but surely
analysing your kernel, and it will eventually complete.

If you wait a very long time and do not see the "Time taken = ..."
message then it could be that GPUVerify is stuck solving a very
difficult set of verification constraints.

In this case, get in touch with the development team and we may be
able to advise you as to how you could annotate your kernel to speed
up analysis.

Here are a couple of general tips:

- Tip 1: add preconditions for scalar parameters

  It might help to add some preconditions to restrict scalar parameters
  of your kernel to specific values.

  For instance, in a matrix transposition kernel:

  ::

    __kernel
    void matrixTranspose(__global float * output,
                         __global float * input,
                         __local  float * block,
                         const    uint    width,
                         const    uint    height,
                         const    uint blockSize
                           )
    {
      ...
    }

  if you know details of the values "width", "height" and "blockSize" should take then you can communicate these to GPUVerify as preconditions, using __requires clauses:

  ::

    __kernel
    void matrixTranspose(__global float * output,
                         __global float * input,
                         __local  float * block,
                         const    uint    width,
                         const    uint    height,
                         const    uint blockSize
                           )
    {
      __requires(blockSize == 16);
      __requires(width == 16*8);
     __requires(height == 16*8);
     ...
    }

  This can, in some cases, dramatically speed up analysis.  Of course,
  specifying specific preconditions means that you are compromising by
  only verifying your kernel for these particular values.


- Tip 2: verify for power-of-two sized values

  GPUVerify tends to do much better during verification when the
  dimensions of work groups and grids are power-of-two sized.
  Similarly, if you find that you have to constrain the values of scalar
  inputs using preconditions then power-of-two sizes are likely to yield
  the best performance.


- Tip 3: scale down local sizes and number of groups

  GPUVerify often succeeds in verifying kernels executed by very large
  numbers of work items.  However, in some cases the difficulty of
  verification is proportional to the number of work items.  If
  verification of a highly parallel kernel is taking a long time then
  you might try verifying for a smaller number of work items first.


**GPUVerify says that my kernel is incorrect, but it's not!**

GPUVerify is a sound static verifier.  A result of this is that the
tool can give "false positives": it may warn you that your kernel can
exhibit a data race or other defect when in fact it cannot.

If this issue is affecting your use of GPUVerify then please get in
touch with the GPUVerify development team.  We can explain how you can
annotate your kernel to allow it to be verified, and we may be able to
extend GPUVerify so that it can handle kernels similar to yours fully
automatically.
