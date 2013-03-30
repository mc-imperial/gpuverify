""" This module defines the paths that GPUVerify will use
    to run the various tools that GPUVerify Depends on.

    These paths must be absolute paths.
"""
import sys

#THIS IS THE TEMPLATE THAT WILL USED FOR DEPLOYING

#The path to the Bugle Source directory. The include-blang/ folder should be in there
bugleSrcDir = sys.path[0] + "/bugle"

#The Path to the directory where the "bugle" executable can be found.
bugleBinDir = sys.path[0] + "/bin"

#The path to the directory where libclc can be found. The nvptex--bugle/ and generic/ folder should be in there
libclcDir = sys.path[0] + "/libclc"

#The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
llvmBinDir = sys.path[0] + "/bin"

#The path to the directory containing GPUVerifyVCGen.exe
gpuVerifyVCGenBinDir = sys.path[0] + "/bin"

#The path to the directory containing GPUVerifyBoogieDriver.exe
gpuVerifyBoogieDriverBinDir = sys.path[0] + "/bin"

#The path to the directory containing z3.exe
z3BinDir = sys.path[0] + "/bin"
