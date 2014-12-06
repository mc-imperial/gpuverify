====================================
Installation
====================================

Getting GPUVerify
=================

Prebuilt versions of GPUVerify are available in two different ways.

* :ref:`docker_containers`
* :ref:`nightly_builds`

.. _docker_containers:

Docker containers
-----------------

`Docker <https://www.docker.com/>`_ is a technology for packaging entire applications into
containers to provide a light-weight (compared to VMs) and portable way to run applications
in an isolated and reproducable environment.

We provide two different images (the difference is each one uses a different SMT solver)
for building GPUVerify containers.

Installing Docker
^^^^^^^^^^^^^^^^^

See this `guide on installing Docker <https://docs.docker.com/installation/#installation>`_ to
learn how to install it.

Z3 image
^^^^^^^^

This container uses Z3 as its SMT solver. Z3 can only be freely used
for non commercial purposes (see https://z3.codeplex.com/license)

To obtain the image run::

    $ docker pull delcypher/gpuverify-docker:z3

CVC4 image
^^^^^^^^^^

This container uses CVC4 as its SMT solver. This version of CVC4 is
built without GPL components so it is under a modified BSD license and so
is usually suitable for commercial use.

To obtain the image run::

    $ docker pull delcypher/gpuverify-docker:cvc4

Running GPUVerify in a container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to get started is to create a new container from the GPUVerify image
you obtained and run an interactive shell in it (replace ``cvc4`` with ``z3`` if you
pulled the ``z3`` image instead of the ``cvc4`` image)::

    $ docker run -ti --rm delcypher/gpuverify-docker:cvc4 /bin/bash

The ``--rm`` flag will remove the container once you exit (don't use this flag if you
want to keep the container).

Inside the container the GPUVerify tool is in your ``PATH`` so you can run for example::

    $ gpuverify --help

The container will have some toy kernels that are part of its testsuite in
``/home/gv/gpuverify/testsuite/`` but you probably want to verify some real
kernels.

To do that you will want to get some kernels into the container so you
can verify them. To do this you can create a volume inside the container that
mounts a directory on the underlying host machine. For example the following
would make the ``/path/to/kernel`` directory visible inside the container as
``/mnt``::

    $ docker run -ti --rm -v /path/to/kernels:/mnt delcypher/gpuverify-docker:cvc4 /bin/bash

Building the GPUVerify image from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Dockerfile``\ s for building our Docker images can be found
in `this GitHub repository <https://github.com/delcypher/gpuverify-docker>`_.

.. _nightly_builds:

Nightly builds
--------------

We build nightly drops for Linux and Windows.
These can be found on the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.

Basic Prequisites
^^^^^^^^^^^^^^^^^

GPUVerify requires python >= 2.7 and the python module `psutil <https://code.google.com/p/psutil/>`_.
On Windows, we recommend installing psutil from a `prebuilt binary <https://pypi.python.org/pypi?:action=display&name=psutil#downloads>`_.
On Linux/OSX, we recommend installing psutil with pip::

     $ pip install psutil

Linux/OSX
^^^^^^^^^
To install GPUVerify follow this guide in a bash shell.

Note ``${INSTALL_ROOT}`` refers to where ever you wish to install GPUVerify.
Replace as appropriate or setup an environment variable.::

     $ export INSTALL_ROOT=/path/to/install

#. Obtain Mono from `<http://www.mono-project.com>`_ and install.

#. Download the Linux 64-bit toolchain zip file from the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.
   Please contact us if you require a 32-bit version.

#. Now unpack the zip file::

      $ cd ${INSTALL_ROOT}
      $ unzip GPUVerifyLinux64-nightly.zip

   This should unpack the GPUVerify toolchain into a path like ``2013-06-11``, which is the date that the tool was packaged.

#. Finally, run the GPUVerify test suite.::

     $ cd ${INSTALL_ROOT}/2013-06-11/gpuverify
     $ ./gvtester.py --write-pickle run.pickle testsuite/

   You should check that your test run matches the current baseline.
   ::

     $ ./gvtester.py --compare-pickle testsuite/baseline.pickle run.pickle

   You should expect the last line of output to be.::

     INFO:testsuite/baseline.pickle = new.pickle

   This means that your install passes the regression suite. 

Windows
^^^^^^^
To install GPUVerify follow this guide in a powershell window.

Note ``${INSTALL_ROOT}`` refers to where ever you wish to build GPUVerify.
Replace as appropriate or setup an environment variable.::

      > ${INSTALL_ROOT}=C:\path\to\install

We recommend that you install GPUVerify to a local hard drive like ``C:``
since this avoids problems with invoking scripts on network mounted
drives.

#. Download the Windows 64-bit toolchain zip file from the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.
   Please contact us if you require a 32-bit version.

#. Right-click on the zip file and select "Properties".
   Now unblock the zip file by clicking on "Unblock" next to "Security".

#. Now unpack the zip file::

      > cd ${INSTALL_ROOT}
      > unzip GPUVerifyWindows64-nightly.zip

   This should unpack the GPUVerify toolchain into a path like ``2013-06-11``, which is the date that the tool was packaged.

#. Finally, run the GPUVerify test suite.::

      > cd ${INSTALL_ROOT}\2013-06-11\gpuverify
      > ./gvtester.py --write-pickle run.pickle testsuite/

   You should check that your test run matches the current baseline.
   ::

      > ./gvtester.py --compare-pickle testsuite/baseline.pickle run.pickle

   You should expect the last line of output to be.::

      INFO:testsuite/baseline.pickle = new.pickle

   This means that your install passes the regression suite. 

