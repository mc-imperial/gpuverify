====================================
Installation
====================================

Getting GPUVerify
==================

We build nightly drops for Linux and Windows.
These can be found on the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.

Linux/OSX
---------
To install GPUVerify follow this guide in a bash shell.

Note ``${INSTALL_ROOT}`` refers to where ever you wish to install GPUVerify.
Replace as appropriate or setup an environment variable.::

     $ export INSTALL_ROOT=/path/to/install

#. (Optional) GPUVerify requires a recent version of `Mono <http://www.mono-project.com>`_ to run.
   You should use a version of Mono >= 3.0.7.
   Here is how to install Mono locally if you do not have it already installed::

      $ cd ${INSTALL_ROOT}
      $ export MONO_VERSION=3.0.7
      $ wget http://download.mono-project.com/sources/mono/mono-${MONO_VERSION}.tar.bz2
      $ tar jxf mono-${MONO_VERSION}.tar.bz2
      $ cd ${INSTALL_ROOT}/mono-${MONO_VERSION}
      $ ./configure --prefix=${INSTALL_ROOT}/local --with-large-heap=yes --enable-nls=no
      $ make
      $ make install

   Now add the Mono binaries to your path.
   You can add this permanently to your ``.bashrc`` or create a ``sourceme.sh`` script to do this automatically::

      $ export PATH=${INSTALL_ROOT}/local/bin:$PATH

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
-------
To install GPUVerify follow this guide in a powershell window.

Note ``${INSTALL_ROOT}`` refers to where ever you wish to build GPUVerify.
Replace as appropriate or setup an environment variable.::

      > ${INSTALL_ROOT}=C:\path\to\install

We recommend that you install GPUVerify to a local hard drive like ``C:``
since this avoids problems with invoking scripts on network mounted
drives.

#. Download the Windows 64-bit toolchain zip file from the GPUVerify `Download <http://multicore.doc.ic.ac.uk/tools/GPUVerify/download.php>`_ page.
   Please contact us if you require a 32-bit version.

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

