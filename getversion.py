import os
import sys
import subprocess
""" This module is responsible for trying to determine the GPUVerify version"""

GPUVerifyDeployVersionFile= os.path.join(
                                          os.path.abspath( os.path.dirname(__file__) ),
                                          '.gvdeployversion'
                                        )
GPUVerifyHgErrorMessage='Error could not retrieve version from Mercurial'


def getVersionStringFromMercurial():
  try:
    p = subprocess.Popen(['hg','log','-r',' -1','--template','Revision {rev}:{node}\n'],
                       stdout=subprocess.PIPE,
                       cwd=sys.path[0])
    (vs, stderr) = p.communicate()

    if p.returncode != 0:
      vs=GPUVerifyHgErrorMessage
  except:
    vs=GPUVerifyHgErrorMessage
  
  return vs

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
    vs += getVersionStringFromMercurial() 
  else:
    vs +="Deployment version\nBuilt from "

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
