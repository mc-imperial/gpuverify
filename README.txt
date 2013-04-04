===BUILDING GPUVERIFY===


Prerequisites:
CMake >=2.8
Python 2.7
Mercurial
Git

Linux/Mac OSX specific prerequisites:
Mono
Make

==Linux/OSX build instructions==

Note ${BUILD_ROOT} refers to where ever you wish to build GPUVerify. Replace
as appropriate

1. Get hold of Clang and LLVM sources (we depend on a specific revision)

$ mkdir -p ${BUILD_ROOT}/llvm_and_clang/src
$ cd ${BUILD_ROOT}/llvm_and_clang/
$ svn  co -r 169118 http://llvm.org/svn/llvm-project/llvm/trunk src
$ cd ${BUILD_ROOT}/llvm_and_clang/src/projects
$ svn co -r 169118 http://llvm.org/svn/llvm-project/cfe/trunk clang
$ svn co -r169118 http://llvm.org/svn/llvm-project/compiler-rt/trunk compiler-rt 
$ cd ${BUILD_ROOT}/llvm_and_clang/src/projects/clang/tools
$ svn co -r169118 http://llvm.org/svn/llvm-project/clang-tools-extra/trunk extra

2. Configure LLVM and Clang for building (we will do an out of source build)

$ mkdir -p ${BUILD_ROOT}/llvm_and_clang/bin
$ cmake -D CMAKE_BUILD_TYPE=Release  ../src_llvm_clang/

Note if you have python3 installed you may need to specifiy
-D PYTHON_EXECUTABLE=/usr/bin/python2.7
to CMake.

If you would like more control over configure process use (cmake-gui or ccmake instead
of cmake).

3. Compile  LLVM and Clang
$ make -jN

where N is the number of jobs to do in parallel.

4. Now get libclc and build
FIX ME!

5. Get Bugle and configure for building (we will do out of source build)

$ mkdir ${BUILD_ROOT}/bugle/
$ cd ${BUILD_ROOT}/bugle/
$ git clone git://git.pcc.me.uk/~peter/bugle.git src
$ mkdir ${BUILD_ROOT}/bugle/bin
$ cd ${BUILD_ROOT}/bugle/bin
$ cmake -D LLVM_CONFIG_EXECUTABLE=${BUILD_ROOT}/llvm_and_clang/bin/bin/llvm-config 
        -D CMAKE_BUILD_TYPE=Release
	-D LIBCLC_DIR=${BUILD_ROOT}/libclc/install

6. Compile Bugle

$ cd ${BUILD_ROOT}/bugle/bin
$ make -jN

where N is the number of jobs to do in parallel

7. Get Z3 (SMT Solver) and build

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

$ make install --ignore-errors

Note that the --ignore-errors option is because the install target will try to
install the Z3 python modules and this will fail if you don't have the correct permissions.
GPUVerify doesn't need the Z3 python modules so we can just ignore this error.

Now we will make a symbolic link because GPUVerify.py looks for "z3.exe" not "z3"
$ cd ${BUILD_ROOT}/z3/install/bin
$ ln -s z3 z3.exe

8. Get GPUVerify code and build C# components

$ cd ${BUILD_ROOT} 
$ hg clone https://hg.codeplex.com/gpuverify 
$ cd ${BUILD_ROOT}/gpuverify
$ xbuild

9. Configure GPUVerify front end.
GPUVerify uses a front end python script (GPUVerify.py). This script needs to be aware of
the location of all its dependencies. We currently do this by having an additional python
script (gvfindtools.py) with hard coded absolute paths that a developer must configure by 
hand. gvfindtools.py is ignored by Mercurial so each developer can their own configuration
without interfering with other users.

$ cd ${BUILD_ROOT}/gpuverify
$ cp gvfindtools.templates/gvfindtools.dev.py gvfindtools.py

Now open gvfindtools.py in your favourite text editor and edit the paths.

If you followed this guide strictly then these paths will be as follows

###############################################################################
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

###############################################################################

10. Run the GPUVerify test suite.

$ cd ${BUILD_ROOT}/gpuverify
$ ./GPUVerifyTester.py --compare-run GPUVerifyTestSuite/baseline.pickle GPUVerifyTestSuite/
