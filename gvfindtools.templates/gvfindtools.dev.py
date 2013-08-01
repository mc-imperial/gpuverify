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
# Please note Windows users should use the following style:
# rootDir = r"c:\projects\gpuverify"
# bugleSrcDir = rootDir + r"\bugle\src"

rootDir = "/home/dan/documents/projects/gpuverify"

#The path to the Bugle Source directory. The include-blang/ folder should be in there
bugleSrcDir = rootDir + "/bugle/src"

#The Path to the directory where the "bugle" executable can be found.
bugleBinDir = rootDir + "/bugle/build"

#The path to the libclc install directory. The include/ and lib/clc/ folders should be there
libclcInstallDir = rootDir + "/libclc/install"

#The path to the llvm Source directory.
llvmSrcDir = rootDir + "/llvm_and_clang/src"

#The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
llvmBinDir = rootDir + "/llvm_and_clang/build/Release/bin"

#The path containing the llvm libraries
llvmLibDir = rootDir + "/llvm_and_clang/build/Release/lib"

#The path to the directory containing GPUVerifyVCGen.exe
gpuVerifyVCGenBinDir = rootDir + "/gpuverify/Binaries"

#The path to the directory containing GPUVerifyBoogieDriver.exe
gpuVerifyBoogieDriverBinDir = rootDir + "/gpuverify/Binaries"

#The path to the z3 Source directory.
z3SrcDir = rootDir + "/z3"

#The path to the directory containing z3.exe
z3BinDir = rootDir + "/z3/build"

#The path to the directory containing cvc4.exe
cvc4BinDir = rootDir + "/cvc4/build"

def init(prefixPath):
  """This method does nothing"""
  pass
