====================================
Installation
====================================

Getting GPUVerify
=================

Prebuilt versions of GPUVerify are available as:

* :ref:`nightly_builds`

.. _nightly_builds:

Nightly builds
--------------

We build nightly drops for Linux and Windows.
These can be found on the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.

Basic Prequisites
^^^^^^^^^^^^^^^^^

GPUVerify requires python >= 2.7 and the python module `psutil <https://code.google.com/p/psutil/>`_.
We recommend installing psutil with pip::

     $ pip install psutil

On Windows, the 32-bit version of the `Visual C++ Redistributable for Visual Studio 2012 <http://www.microsoft.com/en-gb/download/details.aspx?id=30679>`_ is also required. The redistributable will already be installed if a recent version of Visual Studio is present. If not, a version can be obtained for free from Microsoft through the above link. Not having the redistributable installed may lead to crashes.

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

