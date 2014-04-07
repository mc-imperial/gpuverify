""" This module defines the paths that GPUVerify will use
    to run the various tools that GPUVerify Depends on.

    These paths must be absolute paths.
"""
import sys

rootDir = "/home/vagrant/GPUVerify"

# The path to the Bugle Source directory.
# The include-blang/ folder should be there
bugleSrcDir = rootDir + "/bugle/src"

# The Path to the directory where the "bugle" executable can be found.
bugleBinDir = rootDir + "/bugle/build"

# The path to the libclc Source directory.
libclcSrcDir = rootDir + "/libclc/src"

# The path to the libclc install directory.
# The include/ and lib/clc/ folders should be there
libclcInstallDir = rootDir + "/libclc/install"

# The path to the llvm Source directory.
llvmSrcDir = rootDir + "/llvm_and_clang/src"

# The path to the directory containing the llvm binaries.
# llvm-nm, clang and opt should be there
llvmBinDir = rootDir + "/llvm_and_clang/build/Release/bin"

# The path containing the llvm libraries
llvmLibDir = rootDir + "/llvm_and_clang/build/Release/lib"

# The path to the directory containing the GPUVerify binaries.
# GPUVerifyVCGen.exe, GPUVerifyCruncher.exe and GPUVerifyBoogieDriver.exe should be there
gpuVerifyBinDir = rootDir + "/gpuverify/Binaries"

# The path to the z3 Source directory.
z3SrcDir = rootDir + "/z3"

# The path to the directory containing z3.exe
z3BinDir = rootDir + "/z3/build"

# The path to the cvc4 Source directory.
cvc4SrcDir = rootDir + "/CVC4/src"

# The path to the directory containing cvc4.exe
cvc4BinDir = rootDir + "/CVC4/install/bin"

def init(prefixPath):
  """This method does nothing"""
  pass
