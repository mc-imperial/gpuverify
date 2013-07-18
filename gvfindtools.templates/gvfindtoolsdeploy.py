""" This module defines the paths that GPUVerify will use
    to run the various tools that GPUVerify Depends on.

    These paths must be absolute paths.

    Please note that this version of gvfindtools is for deployment ONLY.
    Only modify this is you know what you are doing.
"""
import sys
import os

#Paths, use init() to set them.
bugleSrcDir = ""
bugleBinDir = ""
libclcDir = ""
llvmBinDir = ""
llvmLibDir = ""
gpuVerifyVCGenBinDir = ""
gpuVerifyBoogieDriverBinDir = ""
z3BinDir = ""

def init(pathPrefix):
  """Modify this modules variables by adding a path prefix"""

  global bugleSrcDir, bugleBinDir, libclcDir, llvmBinDir, llvmLibDir, gpuVerifyVCGenBinDir
  global gpuVerifyBoogieDriverBinDir, z3BinDir
  #The path to the Bugle Source directory. The include-blang/ folder should be in there
  bugleSrcDir = pathPrefix + os.sep + "bugle"

  #The Path to the directory where the "bugle" executable can be found.
  bugleBinDir = pathPrefix + os.sep + "bin"

  #The path to the directory where libclc can be found. The nvptx--bugle/ and generic/ folder should be in there
  libclcDir = pathPrefix + os.sep + "libclc"

  #The path to the llvm Source directory. Not used in the deployed setting
  llvmBinDir = pathPrefix + os.sep + "llvm"

  #The path to the directory containing the llvm binaries. llvm-nm, clang and opt should be in there
  llvmBinDir = pathPrefix + os.sep + "bin"

  #The path containing the llvm libraries
  llvmLibDir = pathPrefix + os.sep + "lib"

  #The path to the directory containing GPUVerifyVCGen.exe
  gpuVerifyVCGenBinDir = pathPrefix + os.sep + "bin"

  #The path to the directory containing GPUVerifyBoogieDriver.exe
  gpuVerifyBoogieDriverBinDir = pathPrefix + os.sep + "bin"

  #The path to the z3 Source directory. Not used in the deployed setting
  z3SrcDir = pathPrefix + os.sep + "z3"

  #The path to the directory containing z3.exe
  z3BinDir = pathPrefix + os.sep + "bin"
