#!/usr/bin/env python2.7
import sys
import os
import argparse
import logging
import shutil
import re

#Try to import the paths need for GPUVerify's tools
GPUVerifyRoot = sys.path[0]

if os.path.isfile(sys.path[0] + os.sep + 'gvfindtools.py'):
  import gvfindtools
else:
  sys.stderr.write('Error: Cannot find \'gvfindtools.py\'.'
                   'Did you forget to create it from a template?\n')
  sys.exit(1)

#Import the deploy script
sys.path.insert(0,GPUVerifyRoot + os.sep + 'gvfindtools.templates')
import gvfindtoolsdeploy

class DeployTask:
  def run(self):
    pass
  def makeDestinationDir(self):
    if not os.path.isdir(self.destination):
      os.mkdir(self.destination)

class FileCopy(DeployTask):
  """
      This class is intended for copying individual files
  """
  def __init__(self,srcdir,filename,destination):
    """ 
        srcdir : The directory to copy the file from
        filename : The name of the file in "srcdir"
        destination : The directory to copy to file to
    """
    self.filename = filename
    self.srcpath=srcdir + os.sep + filename
    self.destination=destination

  def run(self):
    if not os.path.isfile(self.srcpath):
      logging.error("Source \"" + self.filename + "\" is not a file.")
      sys.exit(1)

    self.makeDestinationDir()
    logging.info("Copying \"" + self.srcpath + 
                 "\" to \"" + self.destination + "\"")
    shutil.copy(self.srcpath, self.destination)

  def getDestination(self):
    return self.destination + os.sep + self.filename

#This will copy the contents of srcdir into destdir
class DirCopy(DeployTask):
  """
  This Class copies the contents of srcdir into destdir. Note this
  is a recursive copy.

  WARNING if destdir already exists it will be deleted first!
  """
  def __init__(self,srcdir,destdir,copyOnlyRegex=None):
    """
        srcdir        : The directory to copy the contents of
        destdir       : The directory to place the copy of the contents of "srcdir"
        copyOnlyRegex : If not equal to None only filenames that match the regular
                        expression will be copied.
    """
    self.srcdir=srcdir
    self.destination=destdir

    if copyOnlyRegex != None:
      #Construct regex
      self.copyOnlyRegex = re.compile(copyOnlyRegex)
    else:
      self.copyOnlyRegex=None


  def removeDestination(self):
    if os.path.isdir(self.destination):
      logging.warning("Removing directory \"" + self.destination + "\"")
      shutil.rmtree(self.destination)

  def run(self):
    if not os.path.isdir(self.srcdir):
      logging.error("Source directory \"" + self.srcdir + "\" does not exist")
      sys.exit(1)

    #shutil.copytree() requires that the destination folder not already exist
    #this will remove it (be careful!)
    self.removeDestination()

    if self.copyOnlyRegex == None:
      logging.info("Recursively copying \"" + self.srcdir + 
                   "\" into \"" + self.destination + "\"")
      shutil.copytree(self.srcdir,self.destination)
    else:
      logging.info("Recursively copying only files that match \"" + 
                   self.copyOnlyRegex.pattern + "\" from \"" + self.srcdir +
                   "\" into \"" + self.destination + "\"")
      shutil.copytree(self.srcdir,self.destination,ignore=self.listFilesToIgnore)

  def listFilesToIgnore(self,path,filenames):
    logging.debug('Checking ' + path)
    filesToIgnore=[]
    for fileOrDirectory in filenames:
      fullPath=path + os.sep + fileOrDirectory
      if self.copyOnlyRegex.match(fullPath) == None:
        if not os.path.isdir(path + os.sep + fileOrDirectory):
          #The item is a directory and it didn't match the regex
          #so we add it to the ignore list
          filesToIgnore.append(fileOrDirectory)
          logging.debug('ignoring ' + path + os.sep + fileOrDirectory)
      else:
        logging.info('Copying "' + fullPath + '"')
     
    return set(filesToIgnore)


class RegexFileCopy(DeployTask):
  """
      This class will search for all files in a directory
      and if there is a regex match it will be copied into
      the destination directory
  """
  def __init__(self,srcdir,fileRegex,destination,postHook=None):
    """
        srcdir      : The directory to search for files
        fileRegex   : The regex pattern to find files with
        destination : The directory to copy the matched files to
        postHook    : If a file is matched this function will be called.
                      The first argument to the function will the full path
                      to the created file (in "destination").
    """
    self.srcdir=srcdir
    self.fileRegex=fileRegex
    self.destination=destination
    self.postHook=postHook

  def run(self):
    if not os.path.isdir(self.srcdir):
      logging.error("Directory \"" + self.srcdir + "\" does not exist")
      sys.exit(1)
    
    (root, dirs, filenames) = next(os.walk(self.srcdir))

    #compile regex
    logging.info("Searching for files matching regex \"" + self.fileRegex + "\" in \"" + self.srcdir + "\"")
    regex = re.compile(self.fileRegex)
    
    #loop over files in self.srcdir
    for file in filenames:
      if regex.match(file) != None:
        logging.info("\"" + file + "\" matches") 
        action = FileCopy(self.srcdir, file, self.destination)
        action.run()
        #Run the post hook
        if self.postHook != None:
          self.postHook(action.getDestination())

class MoveFile(DeployTask):
  """
      This class is intended to be used to move files.
  """
  def __init__(self,srcpath,destpath):
    """
        srcpath  : The full path to the file to copy
        destpath : The full path to the new location of the file or
                   the directory to move the file to. 
    """
    self.srcpath=srcpath
    self.destpath=destpath

  def run(self):
    if not os.path.isfile(self.srcpath):
      logging.error('File "' + self.srcpath + '" does not exist')

    logging.info('Moving "' + self.srcpath + '" to "' + self.destpath + '"')
    shutil.move(self.srcpath,self.destpath)
     

def main(argv):
  des=('Deploys GPUVerify to a directory by copying the necessary '
      'files from the development directory so that GPUVerify can '
      'be distributed.')
  parser = argparse.ArgumentParser(description=des)
  parser.add_argument("path",
                      help="The path to the directory that GPUVerify will be deployed to"
                     )

  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

  #Check deploy directory exists
  deployDir = args.path
  if not os.path.isdir(deployDir):
    logging.error("Deploy directory \"" + deployDir + "\" doesn't exist.")
    sys.exit(1)

  #Have gvfindtoolsdeploy setup its paths
  deployDir=os.path.abspath(deployDir)
  gvfindtoolsdeploy.init(deployDir)

  #This hook is for Linux/OSX users who build "z3" and not "z3.exe"
  def z3Hook(path):
    if path.endswith('z3'):
      newPath=path + '.exe'
      logging.info('Moving "' + path + '" to "' + newPath + '"')
      shutil.move(path,newPath)

  #Specify actions to perform
  deployActions = [
  DirCopy(gvfindtools.libclcDir, gvfindtoolsdeploy.libclcDir),
  DirCopy(gvfindtools.bugleSrcDir + os.sep + 'include-blang', gvfindtoolsdeploy.bugleSrcDir + os.sep + 'include-blang'),
  FileCopy(GPUVerifyRoot, 'GPUVerify.py', deployDir),
  FileCopy(GPUVerifyRoot, 'gpuverify', deployDir),
  FileCopy(GPUVerifyRoot + os.sep + 'gvfindtools.templates', 'gvfindtoolsdeploy.py', deployDir),
  MoveFile(deployDir + os.sep + 'gvfindtoolsdeploy.py', deployDir + os.sep + 'gvfindtools.py'),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^clang(\.exe)?$', gvfindtoolsdeploy.llvmBinDir ),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^opt(\.exe)?$', gvfindtoolsdeploy.llvmBinDir),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^llvm-nm(\.exe)?$', gvfindtoolsdeploy.llvmBinDir),
  RegexFileCopy(gvfindtools.bugleBinDir, r'bugle(\.exe)?$', gvfindtoolsdeploy.bugleBinDir),
  RegexFileCopy(gvfindtools.gpuVerifyBoogieDriverBinDir, r'^.+\.(dll|exe)$', gvfindtoolsdeploy.gpuVerifyBoogieDriverBinDir),
  FileCopy(gvfindtools.gpuVerifyBoogieDriverBinDir, 'UnivBackPred2.smt2', gvfindtoolsdeploy.gpuVerifyBoogieDriverBinDir),
  RegexFileCopy(gvfindtools.gpuVerifyVCGenBinDir, r'^.+\.(dll|exe)$', gvfindtoolsdeploy.gpuVerifyVCGenBinDir),
  RegexFileCopy(gvfindtools.z3BinDir, r'^z3(\.exe)?$', gvfindtoolsdeploy.z3BinDir, z3Hook),
  DirCopy(gvfindtools.llvmLibDir, gvfindtoolsdeploy.llvmLibDir, copyOnlyRegex=r'^.+\.h$'), # Only Copy clang header files
  FileCopy(GPUVerifyRoot, 'gvtester.py', deployDir),
  DirCopy( os.path.join(GPUVerifyRoot ,'testsuite'), os.path.join(deployDir, 'testsuite') )
  ]

  for action in deployActions:
    action.run()

  logging.info("Deploy finished.")

if __name__ == '__main__':
  main(sys.argv)
