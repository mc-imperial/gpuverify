# vim: set sw=2 ts=2 softtabstop=2 expandtab:
import os
import sys
import gvfindtools
import subprocess
""" This module is responsible for trying to determine the GPUVerify version"""

GPUVerifyDirectory = os.path.abspath( os.path.dirname(__file__))
GPUVerifyDeployVersionFile = os.path.join(GPUVerifyDirectory, '.gvdeployversion')
GPUVerifyRevisionErrorMessage = 'Error getting version information'

def getsha(path, showLocalRev=False):
  oldpath = os.getcwd()
  os.chdir(path)
  if os.path.isdir(os.path.join(path,'.git')):
    if showLocalRev:
      sha = subprocess.check_output(['git','rev-list','HEAD', '--count'])
    else:
      sha = subprocess.check_output(['git','rev-parse','HEAD'])
  elif os.path.isdir(os.path.join(path,'.svn')):
    # The revision number is global for svn
    sha = subprocess.check_output(['svnversion', '-c'])
  elif path == getattr(gvfindtools, 'llvmSrcDir', None):
    sha = subprocess.check_output([getattr(gvfindtools, 'llvmBinDir', None) + '/llvm-config', '--version'])
  else:
    sha = "Error [%s] path is not recognised as a git or svn repository" % path
  os.chdir(oldpath)
  return sha.decode().rstrip('\n\r')

def getVersionStringFromRepos():
  try:
    import gvfindtools
    vs = []

    # This method is used so if a member (e.g. libclcSrcDir)
    # doesn't exist in gvfindtools then we just set None
    # rather than raising an exception.
    def repoTuple(toolName, **kargs ):
      getLocalRev = ( 'getLocalRev' in kargs )

      if 'gvft' in kargs:
        return (toolName, getattr(gvfindtools, kargs['gvft'], None), getLocalRev)
      elif 'path' in kargs:
        return (toolName, kargs['path'], getLocalRev)
      else:
        raise Exception('Misuse of repoTuple')

    for tool, path, localRev in [ repoTuple('llvm', gvft='llvmSrcDir'),
                                  repoTuple('bugle', gvft='bugleSrcDir'),
                                  repoTuple('libclc', gvft='libclcSrcDir'),
                                  repoTuple('vcgen', path=GPUVerifyDirectory),
                                  repoTuple('z3', gvft='z3SrcDir'),
                                  repoTuple('cvc4', gvft='cvc4SrcDir'),
                                  repoTuple('local-revision', path=GPUVerifyDirectory, getLocalRev=True) # GPUVerifyRise4Fun depends on this
                                ]:
      try:
          vs.append(tool.ljust(15) + ": ")
          revision = getsha(path, localRev)
      except Exception as e:
        revision = "No version information available ({0})".format(str(e))

      vs[-1] += revision
    return '\n'.join(vs) + '\n'
  except Exception as e:
    return GPUVerifyRevisionErrorMessage + " : " + str(e)

def getVersionString():
  """
  This will first try to see if we are in a git repo. If so
  version information is retrieved from there.

  If not it will look for a file GPUVerifyDeployVersionFile and if
  it is found it will use that.
  """
  vs="GPUVerify:"

  gitPath = os.path.join(GPUVerifyDirectory, '.git')
  # Look for git
  if os.path.isdir(gitPath):
    vs += " Development version\n"
    vs += getVersionStringFromRepos()
  else:
    vs +=" Deployment version\nBuilt from\n"

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
