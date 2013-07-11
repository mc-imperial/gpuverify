""" This module defines the paths that GPUVerify will use
    to run the various tools that GPUVerify Depends on.

    These paths must be absolute paths.
"""
import sys

# THIS IS A TEMPLATE FOR DEVELOPMENT. MODIFY THE PATHS TO SUIT YOUR BUILD ENVIRONMENT.
# THEN COPY THIS FILE INTO THE ROOT GPUVERIFY DIRECTORY (where GPUVerify.py lives)
# AND RENAME IT TO "gvfindtools.py". "gvfindtools.py" WILL BE IGNORED BY MERCURIAL
# SO IT WILL NOT BE UNDER VERSION CONTROL SO THAT YOU CAN MAINTAIN YOUR OWN PERSONAL
# COPY OF "gvfindtools.py" WITHOUT AFFECTING OTHER DEVELOPERS.
#
# Please note Windows users should use "\\" for paths e.g.
# bugleSrcDir="c:\\build folder\\bugle\\src"

#The path to the Bugle Source directory. The include-blang/ folder should be in there
bugleSrcDir = "/home/dan/documents/projects/gpuverify/bugle/src"

#The Path to the directory where the "bugle" executable can be found.
bugleBinDir = "/home/dan/documents/projects/gpuverify/bugle/bin"

#The path to the directory where libclc can be found. The nvptex--bugle/ and generic/ folders should be in there
libclcDir = "/home/dan/documents/projects/gpuverify/libclc-inst"

#The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
llvmBinDir = "/home/dan/documents/projects/gpuverify/llvm_and_clang/bin/bin"

#The path containing the llvm libraries
llvmLibDir = "/home/dan/documents/projects/gpuverify/llvm_and_clang/bin/lib"

#The path to the directory containing GPUVerifyVCGen.exe
gpuVerifyVCGenBinDir = "/home/dan/documents/projects/gpuverify/gpuverify/GPUVerifyVCGen/bin/Release"

#The path to the directory containing GPUVerifyBoogieDriver.exe
gpuVerifyBoogieDriverBinDir = "/home/dan/documents/projects/gpuverify/gpuverify/GPUVerifyBoogieDriver/bin/Release"

#The path to the directory containing z3.exe
z3BinDir = "/home/dan/documents/projects/gpuverify/z3/install/bin"

#The path to the directory containing cvc4.exe
cvc4BinDir = "/home/dan/documents/projects/gpuverify/cvc4/install/bin"

def init(prefixPath):
  """This method does nothing"""
  pass
