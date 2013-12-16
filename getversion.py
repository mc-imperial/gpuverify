import os
import sys
import subprocess
""" This module is responsible for trying to determine the GPUVerify version"""

GPUVerifyDeployVersionFile= os.path.join(
                                          os.path.abspath( os.path.dirname(__file__) ),
                                          '.gvdeployversion'
                                        )
GPUVerifyRevisionErrorMessage='Error getting version information'

def getsha(path):
  oldpath = os.getcwd()
  os.chdir(path)
  if os.path.isdir(os.path.join(path,'.git')):
    sha = subprocess.check_output(['git','rev-parse','HEAD']).rstrip('\n')
  elif os.path.isdir(os.path.join(path,'.hg')):
    sha = subprocess.check_output(['hg','log','-r','-1','--template','{node}']).rstrip('\n')
  elif os.path.isdir(os.path.join(path,'.svn')):
    sha = subprocess.check_output(['svnversion']).rstrip('\n')
  else:
    sha = "Error [%s] path is not recognised as a git, mercurial or svn repository" % path
  os.chdir(oldpath)
  return sha

def getVersionStringFromRepos():
  try:
    import gvfindtools
    vs = []
    for tool, path in [ ('llvm', gvfindtools.llvmSrcDir),
                        ('bugle', gvfindtools.bugleSrcDir),
                        ('libclc', gvfindtools.libclcSrcDir),
                        ('vcgen', os.path.realpath(__file__)),
                        ('z3', gvfindtools.z3SrcDir),
                        ('cvc4', gvfindtools.cvc4SrcDir) ]:
      if os.path.isdir(path):
        vs.append(tool.ljust(9) + ": " + getsha(path))
    return '\n'.join(vs) + '\n'
  except:
    return GPUVerifyRevisionErrorMessage

def getVersionString():
  """
  This will first try to see if we are in a Mercurial repo. If so
  version information is retrieved from there.

  If not it will look for a file GPUVerifyDeployVersionFile and if
  it is found it will use that.
  """
  vs="GPUVerify:"
  # Look for Mercurial
  if os.path.isdir(sys.path[0] + os.sep + '.hg'):
    vs += "Development version\n"
    vs += getVersionStringFromRepos() 
  else:
    vs +="Deployment version\nBuilt from\n"

    errorMessage = "Error Could not read version from file " + GPUVerifyDeployVersionFile + "\n"
    #Try to open file
    if os.path.isfile(GPUVerifyDeployVersionFile):
      with open(GPUVerifyDeployVersionFile,'r') as f:
        try:
          vs += f.read()
        except IOError:
          vs = errorMessage
    else:
      vs = errorMessage 
  return vs
