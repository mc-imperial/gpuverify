# vim: set sw=2 ts=2 softtabstop=2 expandtab:
import os
import sys
import subprocess
""" This module is responsible for trying to determine the GPUVerify version"""

GPUVerifyDeployVersionFile= os.path.join(
                                          os.path.abspath( os.path.dirname(__file__) ),
                                          '.gvdeployversion'
                                        )
GPUVerifyRevisionErrorMessage='Error getting version information'

def getsha(path, showLocalRev=False):
  oldpath = os.getcwd()
  os.chdir(path)
  if os.path.isdir(os.path.join(path,'.git')):
    if showLocalRev: raise Exception('Not supported')
    sha = subprocess.check_output(['git','rev-parse','HEAD'])

  elif os.path.isdir(os.path.join(path,'.hg')):
    templateKeyword = '{rev}' if showLocalRev else '{node}'
    sha = subprocess.check_output(['hg','log','-r','-1','--template', templateKeyword])

  elif os.path.isdir(os.path.join(path,'.svn')):
    # The revision number is global is svn
    sha = subprocess.check_output(['svnversion'])

  else:
    sha = "Error [%s] path is not recognised as a git, mercurial or svn repository" % path
  os.chdir(oldpath)
  return sha.decode().rstrip('\n')

def getVersionStringFromRepos():
  try:
    import gvfindtools
    vs = []

    # This method is used so if a member (e.g. libclcSrcDir)
    # doesn't exist in gvfindtools then we just set None
    # rather than raising an exception.
    # YUCK: This makes things very inelegant
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
                                  repoTuple('vcgen', path=os.path.dirname(__file__)),
                                  repoTuple('z3', gvft='z3SrcDir'),
                                  repoTuple('cvc4', gvft='cvc4SrcDir'),
                                  repoTuple('local-revision', path=os.path.dirname(__file__), getLocalRev=True) # GPUVerifyRise4Fun depends on this
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
  This will first try to see if we are in a Mercurial repo. If so
  version information is retrieved from there.

  If not it will look for a file GPUVerifyDeployVersionFile and if
  it is found it will use that.
  """
  vs="GPUVerify:"

  hgPath = os.path.join( os.path.dirname(__file__), '.hg')
  # Look for Mercurial
  if os.path.isdir(hgPath):
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
