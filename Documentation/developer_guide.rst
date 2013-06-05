=================================
Developer Guide
=================================

Building GPUVerify
==================

The process for building GPUVerify is rather complicated due to the number of
sub-projects that it depends on. This guide will walk you through the build
process.

There are specific instructions for Linux/OSX and Windows however they have
a common set of prerequisites which are:

* CMake >=2.8
* Python 2.7
* Mercurial
* Git
* Subversion

Linux/OSX
---------
In addition to the common prerequisites a recent version of Mono is also
required due some GPUVerify components being written in C#. You should use a
version of Mono >= 3.0.7.

To build GPUVerify follow these instructions:

Note ``${BUILD_ROOT}`` refers to where ever you wish to build GPUVerify.
Replace as appropriate

.. 
  Note Sphinx is incredibly picky about indentation in lists. Everything
  in the list must be indented aligned with first letter of list text.
  Code blocks must start and end with a blank line and code blocks must be
  further indented from the list text. 

#. Get hold of Clang and LLVM sources (we depend on a specific revision)::

     $ mkdir -p ${BUILD_ROOT}/llvm_and_clang/src
     $ cd ${BUILD_ROOT}/llvm_and_clang/
     $ svn  co -r 169118 http://llvm.org/svn/llvm-project/llvm/trunk src
     $ cd ${BUILD_ROOT}/llvm_and_clang/src/projects
     $ svn co -r 169118 http://llvm.org/svn/llvm-project/cfe/trunk clang
     $ svn co -r169118 http://llvm.org/svn/llvm-project/compiler-rt/trunk compiler-rt 
     $ cd ${BUILD_ROOT}/llvm_and_clang/src/projects/clang/tools
     $ svn co -r169118 http://llvm.org/svn/llvm-project/clang-tools-extra/trunk extra

#. Configure LLVM and Clang for building (we will do an out of source build)::

     $ mkdir -p ${BUILD_ROOT}/llvm_and_clang/bin
     $ cmake -D CMAKE_BUILD_TYPE=Release  ../src_llvm_clang/

   Note if you have python3 installed you may need to specifiy ``-D
   PYTHON_EXECUTABLE=/usr/bin/python2.7`` to CMake.  If you would like more
   control over configure process use (``cmake-gui`` or ``ccmake`` instead of 
   ``cmake``).
#. Compile  LLVM and Clang::

     $ make -jN

   where ``N`` is the number of jobs to do in parallel.
#. Now get libclc and build::

     $ cd ${BUILD_ROOT}
     $ git clone http://llvm.org/git/libclc.git
     $ cd libclc
     $ ./configure.py --with-llvm-config=${BUILD_ROOT}/llvm_and_clang/bin/bin/llvm-config nvptx--bugle
     $ make
#. Get Bugle and configure for building (we will do out of source build)::

     $ mkdir ${BUILD_ROOT}/bugle/
     $ cd ${BUILD_ROOT}/bugle/
     $ git clone git://git.pcc.me.uk/~peter/bugle.git src
     $ mkdir ${BUILD_ROOT}/bugle/bin
     $ cd ${BUILD_ROOT}/bugle/bin
     $ cmake -D LLVM_CONFIG_EXECUTABLE=${BUILD_ROOT}/llvm_and_clang/bin/bin/llvm-config \
             -D CMAKE_BUILD_TYPE=Release \
             -D LIBCLC_DIR=${BUILD_ROOT}/libclc/install
#. Compile Bugle::

    $ cd ${BUILD_ROOT}/bugle/bin
    $ make -jN

   where ``N`` is the number of jobs to do in parallel.
#. Get Z3 (SMT Solver) and build::

    $ mkdir ${BUILD_ROOT}/z3
    $ cd  ${BUILD_ROOT}/z3
    $ mkdir install
    $ git clone https://git01.codeplex.com/z3 src_bin
    $ cd src_bin
    $ ./configure --prefix=${BUILD_ROOT}/z3/install
    $ python2.7 scripts/mk_make.py
    $ cd build
    $ make -jN

   where N is the number of jobs to do in parallel.
   ::

    $ make install --ignore-errors

   Note that the ``--ignore-errors`` option is because the install target will
   try to install the Z3 python modules and this will fail if you don't have the
   correct permissions.  GPUVerify doesn't need the Z3 python modules so we can
   just ignore this error.
   
   Now we will make a symbolic link because ``GPUVerify.py`` looks for ``z3.exe``
   not ``z3``
   ::

    $ cd ${BUILD_ROOT}/z3/install/bin
    $ ln -s z3 z3.exe

#. Get GPUVerify code and build C# components::

     $ cd ${BUILD_ROOT} 
     $ hg clone https://hg.codeplex.com/gpuverify 
     $ cd ${BUILD_ROOT}/gpuverify
     $ xbuild

#. Configure GPUVerify front end.
   GPUVerify uses a front end python script (GPUVerify.py). This script needs
   to be aware of the location of all its dependencies. We currently do this by
   having an additional python script (gvfindtools.py) with hard coded absolute
   paths that a developer must configure by hand. gvfindtools.py is ignored by
   Mercurial so each developer can their own configuration without interfering
   with other users.
   ::

     $ cd ${BUILD_ROOT}/gpuverify
     $ cp gvfindtools.templates/gvfindtools.dev.py gvfindtools.py

   Now open gvfindtools.py in your favourite text editor and edit the paths.
   If you followed this guide strictly then these paths will be as follows
   ::

      #The path to the Bugle Source directory. The include-blang/ folder should be in there
      bugleSrcDir = "${BUILD_ROOT}/bugle/src"

      #The Path to the directory where the "bugle" executable can be found.
      bugleBinDir = "${BUILD_ROOT}/bugle/bin"

      #The path to the directory where libclc can be found. The nvptex--bugle/ and generic/ folders should be in there
      libclcDir = "${BUILD_ROOT}/libclc/install"

      #The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
      llvmBinDir = "${BUILD_ROOT}/llvm_and_clang/bin/bin"

      #The path containing the llvm libraries
      llvmLibDir = "${BUILD_ROOT}/llvm_and_clang/bin/lib"

      #The path to the directory containing GPUVerifyVCGen.exe
      gpuVerifyVCGenBinDir = "${BUILD_ROOT}/gpuverify/GPUVerifyVCGen/bin/Release"

      #The path to the directory containing GPUVerifyBoogieDriver.exe
      gpuVerifyBoogieDriverBinDir = "${BUILD_ROOT}/gpuverify/GPUVerifyBoogieDriver/bin/Release"

      #The path to the directory containing z3.exe
      z3BinDir = "${BUILD_ROOT}/z3/install/bin"

#. Run the GPUVerify test suite.
   ::

     $ cd ${BUILD_ROOT}/gpuverify
     $ ./gvtester.py --write-pickle run.pickle testsuite/

   You can also check that your test run matches the current baseline.
   ::

     $ ./gvtester.py --compare-pickle testsuite/baseline.pickle run.pickle


Windows
-------
**TODO**

Deploying GPUVerify
===================

To deploy a stand alone version of GPUVerify run::

  $ mkdir -p /path/to/deploy/gpuverify
  $ cd ${BUILD_ROOT}/gpuverify
  $ ./deploy.py /path/to/deploy/gpuverify

This will copy the necessary files to run a standalone copy of GPUVerify in an
intelligent manner by 

- Reading ``gvfindtools.py`` to figure out where the 
  dependencies live.
- Reading ``gvfindtools.templates/gvfindtoolsdeploy.py`` to determine
  the directory structure inside the deploy folder.
- Copying ``gvfindtools.templates/gvfindtoolsdeploy.py`` into
  the deploy folder as ``gvfindtools.py`` for ``GPUVerify.py`` to use.

No additional modification of any files is required provided you have correctly
configured your development folder.

Building Boogie
===============

The GPUVerify repository has a pre-built version of Boogie inside it to make
building the project a little bit easier. If you wish to rebuild Boogie for use
in GPUVerify then follow the steps below

**TODO**

Test framework
==============

GPUVerify uses a python script ``gvtester.py`` to instrument the
GPUVerify.py front-end script with a series of tests. These tests are located in
the folder ``testsuite/`` with each test being contained in a seperate
folder.

Test file syntax
----------------

Each test is a file named ``kernel.cu`` or ``kernel.cl`` (for CUDA and OpenCL
respectively). These files contain special comments at the head of the file that
instruct ``gvtester.py`` what to do. The syntax is as follows::


  <line_1> ::= "//" ( "pass" | ("xfail:" <xfail-code> ) )
  <xfail-code> ::= "COMMAND_LINE_ERROR" |
                   "CLANG_ERROR" |
                   "OPT_ERROR" |
                   "BUGLE_ERROR" |
                   "GPUVERIFYVCGEN_ERROR" |
                   "BOOGIE_ERROR" |
                   "BOOGIE_TIMEOUT"

  <line_2> ::= "//" <cmd-args>?
  <cmd-args> ::= <gv-arg> | <gv-arg> " "+ <cmd-args>

  <line_n> ::= "//" <python_regex>

``<line_1>`` is telling ``gvtester.py`` whether or not the kernel is expected
to pass ("pass") or expected to fail ("xfail"). If the kernel is expected to
fail then ``<xfail-code>`` is the expected return code (as a string) from
``GPUVerify.py``.

Note for the most current list of values that ``<xfail-code>`` can take run::

  $ ./gvtester.py --list-xfail-codes


``<line_2>`` is telling ``gvtester.py`` what command line arguments to pass to
``GPUVerify.py``. ``<gv-arg>`` is a single ``GPUVerify.py`` command line
argument. Each command line argument must be seperated by one or more spaces.
Note as stated in the Backus-Naur form it is legal to pass no command line
arguments. The path to the kernel for ``GPUVerify.py`` is implicitly passed as
the last command line argument to ``GPUVerify.py`` so it should **not** be
stated in ``<cmd-args>``.

Special substitution variables can be used inside ``<gv-arg>`` which will
expand as follows:

- ``${KERNEL_DIR}`` : The absolute path to the directory containing the kernel
  without a trailing slash.

``<line_n>`` is telling ``gvtester.py`` what regular expression to match
against the output of ``GPUVerify.py`` if ``GPUVerify.py``'s return code is not
as expected. ``<python_regex>`` is any Python regular expression supported by
the ``re`` module. ``<line_n>`` can be repeated on mulitiple lines. Note that
every character after ``//`` until the end of the line is interpreted as the
regular expression so it is wise to avoid trailing spaces.

Here is a more concrete example

.. code-block:: c++

    //xfail:COMMAND_LINE_ERROR
    //--bad-command-option --boogie-file=${KERNEL_DIR}/axioms.bpl
    //--bad-command-option not recognized\.
    //GPUVerify:[ ]+error:[ ]*
    //GPUVerify: Try --help for list of options

    //This is not a regex because we left a line that did not begin with "//"

    __kernel void hello(__global int* A)
    {
      //...
    }



Pickle format
-------------
``gvtester.py`` is capable of storing information about executed tests in the
"Pickle" format. Use the ``--write-pickle`` option to write a pickle file after
running the tests. This file can be examined using the ``--read-pickle`` option
and the ``--compare-pickles`` option.

Baseline
--------

A pickle file ``testsuite/baseline.pickle`` is provided which should record
``gvtester.py`` being run on ``testsuite/`` in the repository. It is intended
to be a point of reference for developers so they can see if their changes have
broken anything. If you modify something in GPUVerify or add a new test you
should re-generate the baseline.

::

  $ ./gvtester.py --write-pickle ./new-baseline testsuite/
  $ ./gvtester.py -c testsuite/baseline.pickle ./new-baseline

If the comparison looks good and you haven't broken anything then go ahead and
replace the baseline pickle file.

::

  $ mv ./new-baseline testsuite/baseline.pickle

Canonical path prefix
---------------------

When pickle files are generated the full path to each kernel file is recorded.
This could potentially make comparisions (``--compare-pickles``) difficult and
different machines as the absolute paths are likely to be different.

To work around this issue ``gvtester.py`` applies path Canonicalisation
rules to the absolute path to each kernel file when using ``--compare-pickles``.
These rules are:

#. Remove all text leading up to the Canonical path prefix.
#. Replace Windows slashes with UNIX ones.

For example the two paths below refer to the same test. 

- ``/home/person/gpuverify/testsuite/OpenCL/typestest``
- ``c:\program files\gpuverify\testsuite\OpenCL\typestest``

The Canonicalisation rules reduce both of these paths to
``testsuite/OpenCL/typestest`` so they are considered the same test and are
therefore compared.

The default Canonical path prefix is ``testsuite`` but this can be
changed at run time using ``--canonical-path-prefix``.

Adding additional GPUVerify error codes
---------------------------------------

``gvtester.py`` directly imports the GPUVerify codes so that it is aware of the
different error codes that it can return. An additional error condition can
occur where everything passes but one or more regular expressions fail to
match.  ``gvtester.py`` has its own special error code for this which is given
the next available integer after GPUVerify's highest error code. 

This can cause problems if a new error code is added to ``GPUVerify.py`` and
then ``gvtester.py`` is told to examine a pickle file that was generated when
the new error code didn't exist. In this situation ``gvtester.py`` can
incorrectly report the return code of a test. 

For example ``REGEX_MISMATCH_ERROR`` could have the number ``8`` prior to
adding a new error code and a pickle file is recorded that stores the error
code of a particular test as ``8``. Then if a new error code is added, for
example ``WEIRD_ERROR`` then that gets assigned number ``8`` and
``REGEX_MISMATCH_ERROR`` now gets assigned number ``9``.  Now if
``gvtester.py`` opens the old pickle file that contains a test that returned
``8`` then it will report that the test failed with ``WEIRD_ERROR`` instead of
``REGEX_MISMATCH_ERROR`` (which is actually what happened).

If you add new error codes to GPUVerify you should re-generate the baseline
file and be very wary of comparising newly generated pickle files against old
ones.