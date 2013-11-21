=================================
Developer Guide
=================================

Building GPUVerify
==================

The GPUVerify toolchain is a pipeline that uses different components.
This guide will walk you through the build process.

There are specific instructions for Linux, Mac OS X and Windows however they
have a common set of prerequisites which are:

* CMake >=2.8.8
* Python 2.7
* Mercurial
* Git
* Subversion

Linux
-----
In addition to the common prerequisites a Linux build of GPUVerify requires
GCC >= 4.6 and a recent version of Mono since part of the toolchain uses C#.
You should use a version of Mono >= 3.0.7.

To build GPUVerify follow this guide in a bash shell.

Note ``${BUILD_ROOT}`` refers to where ever you wish to build GPUVerify.
Replace as appropriate or setup an environment variable.::

     $ export BUILD_ROOT=/path/to/build

..
  Note Sphinx is incredibly picky about indentation in lists. Everything
  in the list must be indented aligned with first letter of list text.
  Code blocks must start and end with a blank line and code blocks must be
  further indented from the list text.

#. Install Mono::

     $ cd ${BUILD_ROOT}
     $ export MONO_VERSION=3.0.7
     $ wget http://download.mono-project.com/sources/mono/mono-${MONO_VERSION}.tar.bz2
     $ tar jxf mono-${MONO_VERSION}.tar.bz2
     $ cd ${BUILD_ROOT}/mono-${MONO_VERSION}
     $ ./configure --prefix=${BUILD_ROOT}/local --with-large-heap=yes --enable-nls=no
     $ make
     $ make install

   Add the Mono binaries to your path. You can add this permanently to
   your ``.bashrc`` or create a ``sourceme.sh`` script to do this automatically
   ::

     $ export PATH=${BUILD_ROOT}/local/bin:$PATH

#. Get the LLVM and Clang sources (note that GPUVerify depends on LLVM 3.3)::

     $ export LLVM_RELEASE=33
     $ mkdir -p ${BUILD_ROOT}/llvm_and_clang
     $ cd ${BUILD_ROOT}/llvm_and_clang
     $ svn co -q http://llvm.org/svn/llvm-project/llvm/branches/release_${LLVM_RELEASE} src
     $ cd ${BUILD_ROOT}/llvm_and_clang/src/tools
     $ svn co -q http://llvm.org/svn/llvm-project/cfe/branches/release_${LLVM_RELEASE} clang
     $ cd ${BUILD_ROOT}/llvm_and_clang/src/projects
     $ svn co -q http://llvm.org/svn/llvm-project/compiler-rt/branches/release_${LLVM_RELEASE} compiler-rt

   Configure LLVM and Clang for building (we do an out of source build)::

     $ mkdir -p ${BUILD_ROOT}/llvm_and_clang/build
     $ cd ${BUILD_ROOT}/llvm_and_clang/build
     $ cmake -D CMAKE_BUILD_TYPE=Release ../src

   Note if you have python3 installed you may need to specifiy ``-D
   PYTHON_EXECUTABLE=/usr/bin/python2.7`` to CMake.  If you would like to have
   more control over the configure process use (``cmake-gui`` or ``ccmake``
   instead of ``cmake``).

   Compile  LLVM and Clang::

     $ make -jN

   where ``N`` is the number of jobs to run in parallel.

#. Get libclc and build::

     $ cd ${BUILD_ROOT}
     $ git clone http://llvm.org/git/libclc.git ${BUILD_ROOT}/libclc/src
     $ cd libclc/src
     $ ./configure.py --with-llvm-config=${BUILD_ROOT}/llvm_and_clang/build/bin/llvm-config \
                      --prefix=${BUILD_ROOT}/libclc/install \
                      nvptx--bugle
     $ make
     $ make install

#. Get Bugle and configure for building (we do an out of source build)::

     $ cd ${BUILD_ROOT}
     $ git clone git://git.pcc.me.uk/~peter/bugle.git ${BUILD_ROOT}/bugle/src
     $ mkdir ${BUILD_ROOT}/bugle/build
     $ cd ${BUILD_ROOT}/bugle/build
     $ cmake -D LLVM_CONFIG_EXECUTABLE=${BUILD_ROOT}/llvm_and_clang/build/bin/llvm-config \
             -D CMAKE_BUILD_TYPE=Release \
             -D LIBCLC_DIR=${BUILD_ROOT}/libclc/install \
             ../src

   Compile Bugle::

    $ make -jN

   where ``N`` is the number of jobs to run in parallel.

#. Get the Z3 SMT Solver and build::

    $ cd ${BUILD_ROOT}
    $ git clone https://git01.codeplex.com/z3
    $ cd ${BUILD_ROOT}/z3
    $ autoconf
    $ ./configure
    $ python scripts/mk_make.py
    $ cd build
    $ make -jN

   where ``N`` is the number of jobs to run in parallel.

   Make a symbolic link; ``GPUVerify.py`` looks for ``z3.exe`` not ``z3``
   ::

    $ ln -s z3 z3.exe

#. (Optional) Get the CVC4 SMT Solver and build.
   Note that building CVC4 further requires automake and boost::

    $ cd ${BUILD_ROOT}
    $ git clone https://github.com/CVC4/CVC4.git ${BUILD_ROOT}/CVC4/src
    $ cd ${BUILD_ROOT}/CVC4/src
    $ MACHINE_TYPE="x86_64" contrib/get-antlr-3.4
    $ ./autogen.sh
    $ export ANTLR=${BUILD_ROOT}/CVC4/src/antlr-3.4/bin/antlr3
    $ ./configure --with-antlr-dir=${BUILD_ROOT}/CVC4/src/antlr-3.4 \
                  --prefix=${BUILD_ROOT}/CVC4/install \
                  --disable-shared --enable-static
    $ make
    $ make install

   Make a symbolic link; ``GPUVerify.py`` looks for ``cvc4.exe`` not ``cvc4``
   ::

    $ cd ${BUILD_ROOT}/CVC4/install/bin
    $ ln -s cvc4 cvc4.exe

#. Get GPUVerify code and build C# components::

     $ cd ${BUILD_ROOT}
     $ hg clone https://hg.codeplex.com/gpuverify
     $ cd ${BUILD_ROOT}/gpuverify
     $ xbuild /p:Configuration=Release GPUVerify.sln

#. Configure GPUVerify front end.
   GPUVerify uses a front end python script (GPUVerify.py). This script needs
   to be aware of the location of all its dependencies. We currently do this by
   having an additional python script (gvfindtools.py) with hard coded absolute
   paths that a developer must configure by hand. gvfindtools.py is ignored by
   Mercurial so each developer can have their own configuration without
   interfering with other users.
   ::

     $ cd ${BUILD_ROOT}/gpuverify
     $ cp gvfindtools.templates/gvfindtools.dev.py gvfindtools.py

   Open gvfindtools.py in a text editor and edit the paths.
   If you followed this guide strictly then these paths will be as follows
   and you should only need to change the ``rootDir`` variable.
   ::

      rootDir = "${BUILD_ROOT}" #< CHANGE THIS PATH

      #The path to the Bugle Source directory. The include-blang/ folder should be in there
      bugleSrcDir = rootDir + "/bugle/src"

      #The Path to the directory where the "bugle" executable can be found.
      bugleBinDir = rootDir + "/bugle/build"

      #The path to the libclc Source directory.
      libclcSrcDir = rootDir + "/libclc/src"

      #The path to the libclc install directory. The include/ and lib/clc/ folders should be there
      libclcInstallDir = rootDir + "/libclc/install"

      #The path to the llvm Source directory.
      llvmSrcDir = rootDir + "/llvm_and_clang/src"

      #The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
      llvmBinDir = rootDir + "/llvm_and_clang/build/bin"

      #The path containing the llvm libraries
      llvmLibDir = rootDir + "/llvm_and_clang/build/lib"

      #The path to the directory containing GPUVerifyVCGen.exe
      gpuVerifyVCGenBinDir = rootDir + "/gpuverify/Binaries"

      #The path to the directory containing GPUVerifyCruncher.exe
      gpuVerifyCruncherBinDir = rootDir + "/gpuverify/Binaries"

      #The path to the directory containing GPUVerifyBoogieDriver.exe
      gpuVerifyBoogieDriverBinDir = rootDir + "/gpuverify/Binaries"

      #The path to the z3 Source directory.
      z3SrcDir = rootDir + "/z3"

      #The path to the directory containing z3.exe
      z3BinDir = rootDir + "/z3/build"

      #The path to the cvc4 Source directory.
      cvc4SrcDir = rootDir + "/CVC4/src"

      #The path to the directory containing cvc4.exe
      cvc4BinDir = rootDir + "/CVC4/install/bin"

#. (Optional) Build the documentation. This requires the Sphinx python module,
   which you can install using ``easy_install``.::

    $ easy_install Sphinx
    $ cd ${BUILD_ROOT}/gpuverify/Documentation
    $ make html

#. Run the GPUVerify test suite.
   ::

     $ cd ${BUILD_ROOT}/gpuverify
     $ ./gvtester.py --write-pickle run.pickle testsuite

   To run the GPUVerify test suite using the CVC4 SMT Solver:
   ::

     $ ./gvtester.py --gvopt="--solver=cvc4" --write-pickle run.pickle testsuite

   You can also check that your test run matches the current baseline.
   ::

     $ ./gvtester.py --compare-pickle testsuite/baseline.pickle run.pickle

   You can also check that your CVC4 test run matches the current CVC4 baseline.
   ::

     $ ./gvtester.py --compare-pickle testsuite/baseline_cvc4.pickle run.pickle

   You should expect the last line of output to be.::

     INFO:testsuite/baseline.pickle = new.pickle

   This means that your install passes the regression suite.

Mac OS X
--------
In addition to the common prerequisites a Mac build of GPUVerify requires
a recent version of Mono since part of the toolchain uses C#.
You should use a version of Mono >= 3.0.7.

To build GPUVerify follow this guide in a bash shell.

Note ``${BUILD_ROOT}`` refers to where ever you wish to build GPUVerify.
Replace as appropriate or setup an environment variable.::

     $ export BUILD_ROOT=/path/to/build

#. Obtain Mono from `<http://www.mono-project.com>`_ and install.

#. Get the LLVM and Clang sources (note that GPUVerify depends on LLVM 3.3)::

     $ export LLVM_RELEASE=33
     $ mkdir -p ${BUILD_ROOT}/llvm_and_clang
     $ cd ${BUILD_ROOT}/llvm_and_clang
     $ svn co -q http://llvm.org/svn/llvm-project/llvm/branches/release_${LLVM_RELEASE} src
     $ cd ${BUILD_ROOT}/llvm_and_clang/src/tools
     $ svn co -q http://llvm.org/svn/llvm-project/cfe/branches/release_${LLVM_RELEASE} clang
     $ cd ${BUILD_ROOT}/llvm_and_clang/src/projects
     $ svn co -q http://llvm.org/svn/llvm-project/compiler-rt/branches/release_${LLVM_RELEASE} compiler-rt

   Configure LLVM and Clang for building (we do an out of source build)::

     $ mkdir -p ${BUILD_ROOT}/llvm_and_clang/build
     $ cd ${BUILD_ROOT}/llvm_and_clang/build
     $ CXXFLAGS="-stdlib=libc++" LDFLAGS="-stdlib=libc++" \
       cmake -D CMAKE_BUILD_TYPE=Release ../src

   Compile  LLVM and Clang::

     $ make -jN

   where ``N`` is the number of jobs to run in parallel.

#. Get libclc and build::

     $ cd ${BUILD_ROOT}
     $ git clone http://llvm.org/git/libclc.git ${BUILD_ROOT}/libclc/src
     $ cd libclc/src
     $ ./configure.py --with-llvm-config=${BUILD_ROOT}/llvm_and_clang/build/bin/llvm-config \
                      --with-cxx-compiler=c++ \
                      --prefix=${BUILD_ROOT}/libclc/install \
                      nvptx--bugle
     $ make
     $ make install

#. Get Bugle and configure for building (we do an out of source build)::

     $ cd ${BUILD_ROOT}
     $ git clone git://git.pcc.me.uk/~peter/bugle.git ${BUILD_ROOT}/bugle/src
     $ mkdir ${BUILD_ROOT}/bugle/build
     $ cd ${BUILD_ROOT}/bugle/build
     $ cmake -D LLVM_CONFIG_EXECUTABLE=${BUILD_ROOT}/llvm_and_clang/build/bin/llvm-config \
             -D CMAKE_BUILD_TYPE=Release \
             -D LIBCLC_DIR=${BUILD_ROOT}/libclc/install \
             ../src

   Compile Bugle::

    $ make -jN

   where ``N`` is the number of jobs to run in parallel.

#. Get the Z3 SMT Solver and build::

    $ cd ${BUILD_ROOT}
    $ git clone https://git01.codeplex.com/z3
    $ cd ${BUILD_ROOT}/z3
    $ autoconf
    $ ./configure
    $ python scripts/mk_make.py
    $ cd build
    $ make -jN

   where ``N`` is the number of jobs to run in parallel.

   Make a symbolic link; ``GPUVerify.py`` looks for ``z3.exe`` not ``z3``
   ::

    $ ln -s z3 z3.exe

#. (Optional) Get the CVC4 SMT Solver and build.
   Note that building CVC4 further requires automake and boost::

    $ cd ${BUILD_ROOT}
    $ git clone https://github.com/CVC4/CVC4.git ${BUILD_ROOT}/CVC4/src
    $ cd ${BUILD_ROOT}/CVC4/src
    $ MACHINE_TYPE="x86_64" contrib/get-antlr-3.4
    $ ./autogen.sh
    $ export ANTLR=${BUILD_ROOT}/CVC4/src/antlr-3.4/bin/antlr3
    $ ./configure --with-antlr-dir=${BUILD_ROOT}/CVC4/src/antlr-3.4 \
                  --prefix=${BUILD_ROOT}/CVC4/install \
                  --disable-shared --enable-static
    $ make
    $ make install

   Make a symbolic link; ``GPUVerify.py`` looks for ``cvc4.exe`` not ``cvc4``
   ::

    $ cd ${BUILD_ROOT}/CVC4/install/bin
    $ ln -s cvc4 cvc4.exe

   Note that if CVC4 needs to be deployed to a system different from the one
   on which it is being built, the GMP libraries on the build system need to
   be static and not dynamic.

#. Get GPUVerify code and build C# components::

     $ cd ${BUILD_ROOT}
     $ hg clone https://hg.codeplex.com/gpuverify
     $ cd ${BUILD_ROOT}/gpuverify
     $ xbuild /p:Configuration=Release GPUVerify.sln

#. Configure GPUVerify front end.
   GPUVerify uses a front end python script (GPUVerify.py). This script needs
   to be aware of the location of all its dependencies. We currently do this by
   having an additional python script (gvfindtools.py) with hard coded absolute
   paths that a developer must configure by hand. gvfindtools.py is ignored by
   Mercurial so each developer can have their own configuration without
   interfering with other users.
   ::

     $ cd ${BUILD_ROOT}/gpuverify
     $ cp gvfindtools.templates/gvfindtools.dev.py gvfindtools.py

   Open gvfindtools.py in a text editor and edit the paths.
   If you followed this guide strictly then these paths will be as follows
   and you should only need to change the ``rootDir`` variable.
   ::

      rootDir = "${BUILD_ROOT}" #< CHANGE THIS PATH

      #The path to the Bugle Source directory. The include-blang/ folder should be in there
      bugleSrcDir = rootDir + "/bugle/src"

      #The Path to the directory where the "bugle" executable can be found.
      bugleBinDir = rootDir + "/bugle/build"

      #The path to the libclc Source directory.
      libclcSrcDir = rootDir + "/libclc/src"

      #The path to the libclc install directory. The include/ and lib/clc/ folders should be there
      libclcInstallDir = rootDir + "/libclc/install"

      #The path to the llvm Source directory.
      llvmSrcDir = rootDir + "/llvm_and_clang/src"

      #The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
      llvmBinDir = rootDir + "/llvm_and_clang/build/bin"

      #The path containing the llvm libraries
      llvmLibDir = rootDir + "/llvm_and_clang/build/lib"

      #The path to the directory containing GPUVerifyVCGen.exe
      gpuVerifyVCGenBinDir = rootDir + "/gpuverify/Binaries"

      #The path to the directory containing GPUVerifyCruncher.exe
      gpuVerifyCruncherBinDir = rootDir + "/gpuverify/Binaries"

      #The path to the directory containing GPUVerifyBoogieDriver.exe
      gpuVerifyBoogieDriverBinDir = rootDir + "/gpuverify/Binaries"

      #The path to the z3 Source directory.
      z3SrcDir = rootDir + "/z3"

      #The path to the directory containing z3.exe
      z3BinDir = rootDir + "/z3/build"

      #The path to the cvc4 Source directory.
      cvc4SrcDir = rootDir + "/CVC4/src"

      #The path to the directory containing cvc4.exe
      cvc4BinDir = rootDir + "/CVC4/install/bin"

#. (Optional) Build the documentation. This requires the Sphinx python module,
   which you can install using ``easy_install``.::

    $ easy_install Sphinx
    $ cd ${BUILD_ROOT}/gpuverify/Documentation
    $ make html

#. Run the GPUVerify test suite.
   ::

     $ cd ${BUILD_ROOT}/gpuverify
     $ ./gvtester.py --write-pickle run.pickle testsuite

   To run the GPUVerify test suite using the CVC4 SMT Solver:
   ::

     $ ./gvtester.py --gvopt="--solver=cvc4" --write-pickle run.pickle testsuite

   You can also check that your test run matches the current baseline.
   ::

     $ ./gvtester.py --compare-pickle testsuite/baseline.pickle run.pickle

   You can also check that your CVC4 test run matches the current CVC4 baseline.
   ::

     $ ./gvtester.py --compare-pickle testsuite/baseline_cvc4.pickle run.pickle

   You should expect the last line of output to be.::

     INFO:testsuite/baseline.pickle = new.pickle

   This means that your install passes the regression suite.

Windows
-------
In addition to the common prerequisites a Windows build of GPUVerify requires
Microsoft Visual Studio 2010 and GnuWin32.

To build GPUVerify follow this guide in a powershell window.

Note ``${BUILD_ROOT}`` refers to where ever you wish to build GPUVerify.
Replace as appropriate or setup an environment variable.::

      > ${BUILD_ROOT}='C:\path\to\build'

We recommend that you build GPUVerify to a local hard drive like ``C:``
since this avoids problems with invoking scripts on network mounted
drives.

#. (Optional) Setup Microsoft Visual Studio tools for your shell.
   This will enable you to build projects from the command line.::

      pushd 'C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC'
      cmd /c "vcvarsall.bat&set" | foreach {
        if ($_ -match "=") {
          $v = $_.split("="); set-item -force -path "ENV:\$($v[0])" -value "$($v[1])"
        }
      }
      popd

   You can add this permanently to your ``$profile`` so that the Microsoft
   compiler is always available at the command-line.

#. Get the LLVM and Clang sources (note that GPUVerify depends LLVM 3.3)::

      > $LLVM_RELEASE=33
      > mkdir ${BUILD_ROOT}\llvm_and_clang
      > cd ${BUILD_ROOT}\llvm_and_clang
      > svn co -q http://llvm.org/svn/llvm-project/llvm/branches/release_$LLVM_RELEASE src
      > cd ${BUILD_ROOT}\llvm_and_clang\src\tools
      > svn co -q http://llvm.org/svn/llvm-project/cfe/branches/release_$LLVM_RELEASE clang
      > cd ${BUILD_ROOT}\llvm_and_clang\src\projects
      > svn co -q http://llvm.org/svn/llvm-project/compiler-rt/branches/release_$LLVM_RELEASE compiler-rt

   Configure LLVM and Clang for building (we do an out of source build)::

      > mkdir ${BUILD_ROOT}\llvm_and_clang\build
      > cd ${BUILD_ROOT}\llvm_and_clang\build
      > cmake -D CMAKE_BUILD_TYPE=Release ../src

   Compile LLVM and Clang. You can do this by opening ``LLVM.sln`` in Visual
   Studio and building, or alternatively, if you have setup the Microsoft tools
   for the command line, then::

      > msbuild /p:Configuration=Release LLVM.sln

#. Get libclc source and binaries. You can download the binaries from the
   GPUVerify website and unzip this in ``${BUILD_ROOT}``. From the command
   line do::

      > cd ${BUILD_ROOT}
      > mkdir ${BUILD_ROOT}\libclc
      > git clone http://llvm.org/git/libclc.git ${BUILD_ROOT}\libclc\src
      > $libclc_url = "http://multicore.doc.ic.ac.uk/tools/downloads/libclc-nightly.zip"
      > (new-object System.Net.WebClient).DownloadFile($libclc_url, "${BUILD_ROOT}\libclc-nightly.zip")
      > $shell = new-object -com shell.application
      > $zip   = $shell.namespace("${BUILD_ROOT}\libclc-nightly.zip")
      > $dest  = $shell.namespace("${BUILD_ROOT}")
      > $dest.Copyhere($zip.items(), 0x14)
      > del ${BUILD_ROOT}\libclc-nightly.zip

#. Get Bugle and configure for building (we do an out of source build)::

      > cd ${BUILD_ROOT}
      > mkdir ${BUILD_ROOT}\bugle
      > git clone git://git.pcc.me.uk/~peter/bugle.git ${BUILD_ROOT}\bugle\src
      > mkdir ${BUILD_ROOT}\bugle\build
      > cd ${BUILD_ROOT}\bugle\build
      > $LLVM_SRC = "${BUILD_ROOT}\llvm_and_clang\src"
      > $LLVM_BUILD = "${BUILD_ROOT}\llvm_and_clang\build"
      > cmake -G "Visual Studio 10" `
              -D LLVM_SRC=$LLVM_SRC `
              -D LLVM_BUILD=$LLVM_BUILD `
              -D LLVM_BUILD_TYPE=Release `
              -D LIBCLC_DIR=${BUILD_ROOT}\libclc\install `
              ..\src

   Compile Bugle. You can do this by opening ``Bugle.sln`` in Visual
   Studio and building, or alternatively, if you have setup the Microsoft tools
   for the command line, then::

      > msbuild /p:Configuration=Release Bugle.sln

#. Get the Z3 SMT Solver and build::

      > cd ${BUILD_ROOT}
      > git clone https://git01.codeplex.com/z3
      > cd ${BUILD_ROOT}}\z3
      > python scripts\mk_make.py
      > cd build
      > nmake

#. (Optional) Get the CVC4 SMT Solver::

      > cd ${BUILD_ROOT}
      > mkdir -p ${BUILD_ROOT}\cvc4\build
      > cd ${BUILD_ROOT}\cvc4\build
      > $cvc4_url = "http://cvc4.cs.nyu.edu/builds/win32-opt/unstable/cvc4-2013-07-20-win32-opt.exe"
      > (new-object System.Net.WebClient).DownloadFile($cvc4_url, "${BUILD_ROOT}\cvc4\build\cvc4.exe")

#. Get GPUVerify code and build. You can do this by opening ``GPUVerify.sln``
   in Visual Studio and building, or alternatively, if you have setup the
   Microsoft tools for the command line, then::

      > cd ${BUILD_ROOT}
      > hg clone https://hg.codeplex.com/gpuverify
      > cd ${BUILD_ROOT}\gpuverify
      > msbuild /p:Configuration=Release GPUVerify.sln

#. Configure GPUVerify front end.::

     > cd ${BUILD_ROOT}\gpuverify
     > copy gvfindtools.templates\gvfindtools.dev.py gvfindtools.py

   Open gvfindtools.py in a text editor and edit the paths.
   If you followed this guide strictly then these paths will be as follows
   and you should only need to change the ``rootDir`` variable.
   ::

      rootDir = r"${BUILD_ROOT}" #< CHANGE THIS PATH

      #The path to the Bugle Source directory. The include-blang/ folder should be in there
      bugleSrcDir = rootDir + r"\bugle\src"

      #The Path to the directory where the "bugle" executable can be found.
      bugleBinDir = rootDir + r"\bugle\build\Release"

      #The path to the libclc Source directory.
      libclcSrcDir = rootDir + r"\libclc\src"

      #The path to the libclc install directory. The include/ and lib/clc/ folders should be there
      libclcInstallDir = rootDir + r"\libclc\install"

      #The path to the llvm Source directory.
      llvmSrcDir = rootDir + r"\llvm_and_clang\src"

      #The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
      llvmBinDir = rootDir + r"\llvm_and_clang\build\bin\Release"

      #The path containing the llvm libraries
      llvmLibDir = rootDir + r"\llvm_and_clang\build\lib"

      #The path to the directory containing GPUVerifyVCGen.exe
      gpuVerifyVCGenBinDir = rootDir + r"\gpuverify\Binaries"

      #The path to the directory containing GPUVerifyCruncher.exe
      gpuVerifyCruncherBinDir = rootDir + r"\gpuverify\Binaries"

      #The path to the directory containing GPUVerifyBoogieDriver.exe
      gpuVerifyBoogieDriverBinDir = rootDir + r"\gpuverify\Binaries"

      #The path to the z3 Source directory.
      z3SrcDir = rootDir + r"\z3"

      #The path to the directory containing z3.exe
      z3BinDir = rootDir + r"\z3\build"

      #The path to the directory containing cvc4.exe
      cvc4BinDir = rootDir + r"\cvc4\build"

#. (Optional) Build the documentation. This requires the Sphinx python module,
   which you can install using ``easy_install``.::

    $ easy_install Sphinx
    $ cd ${BUILD_ROOT}\gpuverify\Documentation
    $ make html

#. Run the GPUVerify test suite.
   ::

     $ cd ${BUILD_ROOT}\gpuverify
     $ .\gvtester.py --write-pickle run.pickle testsuite

   To run the GPUVerify test suite using the CVC4 SMT Solver:
   ::

     $ .\gvtester.py --gvopt="--solver=cvc4" --write-pickle run.pickle testsuite

   You can also check that your test run matches the current baseline.
   ::

     $ .\gvtester.py --compare-pickle testsuite\baseline.pickle run.pickle

   You can also check that your CVC4 test run matches the current CVC4 baseline.
   ::

     $ .\gvtester.py --compare-pickle testsuite\baseline_cvc4.pickle run.pickle

   You should expect the last line of output to be::

     INFO:testsuite/baseline.pickle = new.pickle

   This means that your install passes the regression suite.

Deploying GPUVerify
===================

To deploy a stand alone version of GPUVerify run::

  $ mkdir -p /path/to/deploy/gpuverify
  $ cd ${BUILD_ROOT}/gpuverify
  $ ./deploy.py /path/to/deploy/gpuverify

In the case you only built the Z3 solver, additionally supply the
``--solver=z3`` option to ``deploy.py``.

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
in GPUVerify then follow the steps below for Linux and Mac OS X.::

      $ cd ${BUILD_ROOT}
      $ hg clone https://hg.codeplex.com/boogie
      $ cd boogie
      $ xbuild /p:TargetFrameworkProfile="" /p:Configuration=Release Boogie.sln
      $ cd Binaries
      $ ls ${BUILD_ROOT}/gpuverify/BoogieBinaries \
             | xargs -I{} -t cp {} ${BUILD_ROOT}/gpuverify/BoogieBinaries

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


  <line_1>     ::= "//" ( "pass" | ("xfail:" <xfail-code> ) )
  <xfail-code> ::= "COMMAND_LINE_ERROR"
                |  "CLANG_ERROR"
                |  "OPT_ERROR"
                |  "BUGLE_ERROR"
                |  "GPUVERIFYVCGEN_ERROR"
                |  "CRUNCHER_ERROR"
                |  "BOOGIE_ERROR"
                |  "TIMEOUT"
                |  "CTRL_C"

  <line_2>     ::= "//" <cmd-args>?
  <cmd-args>   ::= <gv-arg> | <gv-arg> " "+ <cmd-args>

  <line_n>     ::= "//" <python_regex>

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
``gvtester.py`` being run on ``testsuite`` in the repository. It is intended
to be a point of reference for developers so they can see if their changes have
broken anything. If you modify something in GPUVerify or add a new test you
should re-generate the baseline.

::

  $ ./gvtester.py --write-pickle ./new-baseline testsuite
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
different error codes that it can return. An additional error condition
(REGEX_MISMATCH_ERROR) can occur where everything passes but one or more
regular expressions fail to match.  ``gvtester.py`` has its own special error
code for this. At run time ``gvtester.py`` will check there is no conflict
between the GPUVerify error codes and REGEX_MISMATCH_ERROR.

To add an error code simply add it to the ErrorCodes class in ``GPUVerify.py``.
Make sure your new error code has a value larger than existing error codes.
There is no need to regenerate the baseline unless you've changed the testsuite
in some way.
