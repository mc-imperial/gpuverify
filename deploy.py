#!/usr/bin/env python2.7
import sys
import os
import argparse
import logging
import shutil
import re
import datetime

import getversion

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

class IfUsing(DeployTask):
  """ Wrapper class that only executes a deploy task if host is of
      particular type.
  """

  """ Set the task to execute is host operating system matches 'operatingSystem'.
      Valid values for 'operatingSystem' are those of os.name
  """
  def __init__(self,operatingSystem,task):
    self.task=task
    self.operatingSystem=operatingSystem
  def run(self):
    if os.name == self.operatingSystem:
      logging.info("Using " + self.operatingSystem + ", performing task")
      self.task.run()
    else:
      logging.info("Not using " + self.operatingSystem + ", skipping task")


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

class CreateFileFromString(DeployTask):
  """
      This class will create a text file from a string
  """

  def __init__(self,string,destpath):
    self.string=string
    self.destpath=destpath

  def run(self):
    logging.info('Creating file "' + self.destpath + '"')

    with open(self.destpath,'w') as f:
      f.write(self.string)

def main(argv):
  des=('Deploys GPUVerify to a directory by copying the necessary '
      'files from the development directory so that GPUVerify can '
      'be distributed.')
  parser = argparse.ArgumentParser(description=des)
  parser.add_argument("path",
                      help = "The path to the directory that GPUVerify will be deployed to"
                     )
  parser.add_argument("--quiet",
                      help = "only output errors",
                      action = "store_true",
                      default = False
                     )
  parser.add_argument("--solver",
                      help = "solvers to include in deployment (all, z3, cvc4)",
                      type = str,
                      default = 'all'
                     )

  args = parser.parse_args()
  level=logging.INFO
  if args.quiet:
    level=logging.ERROR
  logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

  #Check solvers
  if args.solver not in ['all','z3','cvc4']:
    logging.error("Solver must be one of all, z3 or cvc4")
    sys.exit(1)

  #Check deploy directory exists
  deployDir = args.path
  if not os.path.isdir(deployDir):
    logging.error("Deploy directory \"" + deployDir + "\" doesn't exist.")
    sys.exit(1)

  #Have gvfindtoolsdeploy setup its paths
  deployDir=os.path.abspath(deployDir)
  gvfindtoolsdeploy.init(deployDir)

  #License path and string
  licenseDest = os.path.join(deployDir, 'licenses')
  licenseString = "Licenses can be found in the license directory\n"

  #Determine version and create version string
  versionString = getversion.getVersionStringFromMercurial()
  versionString += "Deployed on " + datetime.datetime.utcnow().ctime() + " (UTC)"

  #Specify actions to perform
  deployActions = [
  # libclc
  DirCopy(gvfindtools.libclcInstallDir, gvfindtoolsdeploy.libclcInstallDir),
  FileCopy(gvfindtools.libclcSrcDir, 'LICENSE.TXT', licenseDest),
  MoveFile(licenseDest + os.sep + 'LICENSE.TXT', licenseDest + os.sep + 'libclc.txt'),
  # bugle
  DirCopy(gvfindtools.bugleSrcDir + os.sep + 'include-blang', gvfindtoolsdeploy.bugleSrcDir + os.sep + 'include-blang'),
  FileCopy(gvfindtools.bugleSrcDir, 'LICENSE.TXT', licenseDest),
  MoveFile(licenseDest + os.sep + 'LICENSE.TXT', licenseDest + os.sep + 'bugle.txt'),
  RegexFileCopy(gvfindtools.bugleBinDir, r'bugle(\.exe)?$', gvfindtoolsdeploy.bugleBinDir),
  RegexFileCopy(gvfindtools.bugleBinDir, r'libbugleInlineCheckPlugin\.(so|dylib)?$', gvfindtoolsdeploy.bugleBinDir),
  # GPUVerify
  FileCopy(GPUVerifyRoot, 'GPUVerify.py', deployDir),
  FileCopy(GPUVerifyRoot, 'getversion.py', deployDir),
  FileCopy(GPUVerifyRoot, 'LICENSE.TXT', licenseDest),
  MoveFile(licenseDest + os.sep + 'LICENSE.TXT', licenseDest + os.sep + 'gpuverify-boogie.txt'),
  IfUsing('posix',FileCopy(GPUVerifyRoot, 'gpuverify', deployDir)),
  IfUsing('nt',FileCopy(GPUVerifyRoot, 'GPUVerify.bat', deployDir)),
  FileCopy(GPUVerifyRoot + os.sep + 'gvfindtools.templates', 'gvfindtoolsdeploy.py', deployDir),
  MoveFile(deployDir + os.sep + 'gvfindtoolsdeploy.py', deployDir + os.sep + 'gvfindtools.py'),
  RegexFileCopy(gvfindtools.gpuVerifyBoogieDriverBinDir, r'^.+\.(dll|exe)$', gvfindtoolsdeploy.gpuVerifyBoogieDriverBinDir),
  FileCopy(gvfindtools.gpuVerifyBoogieDriverBinDir, 'UnivBackPred2.smt2', gvfindtoolsdeploy.gpuVerifyBoogieDriverBinDir),
  RegexFileCopy(gvfindtools.gpuVerifyVCGenBinDir, r'^.+\.(dll|exe)$', gvfindtoolsdeploy.gpuVerifyVCGenBinDir),
  FileCopy(GPUVerifyRoot, 'gvtester.py', deployDir),
  DirCopy(os.path.join(GPUVerifyRoot ,'testsuite'), os.path.join(deployDir, 'testsuite')),
  # llvm, clang
  FileCopy(gvfindtools.llvmSrcDir, 'LICENSE.TXT', licenseDest),
  MoveFile(licenseDest + os.sep + 'LICENSE.TXT', licenseDest + os.sep + 'llvm.txt'),
  FileCopy(os.path.join(gvfindtools.llvmSrcDir, 'tools' + os.sep + 'clang'), 'LICENSE.TXT', licenseDest),
  MoveFile(licenseDest + os.sep + 'LICENSE.TXT', licenseDest + os.sep + 'clang.txt'),
  FileCopy(os.path.join(gvfindtools.llvmSrcDir, 'projects' + os.sep + 'compiler-rt'), 'LICENSE.TXT', licenseDest),
  MoveFile(licenseDest + os.sep + 'LICENSE.TXT', licenseDest + os.sep + 'compiler-rt.txt'),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^clang(\.exe)?$', gvfindtoolsdeploy.llvmBinDir ),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^opt(\.exe)?$', gvfindtoolsdeploy.llvmBinDir),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^llvm-nm(\.exe)?$', gvfindtoolsdeploy.llvmBinDir),
  DirCopy(gvfindtools.llvmLibDir, gvfindtoolsdeploy.llvmLibDir, copyOnlyRegex=r'^.+\.h$'), # Only Copy clang header files
  # file for version information
  CreateFileFromString(versionString, os.path.join(deployDir, os.path.basename(getversion.GPUVerifyDeployVersionFile))),
  # license file
  CreateFileFromString(licenseString, os.path.join(deployDir, "LICENSE.TXT"))
  ]

  if args.solver in ['all','z3']:
    deployActions.extend([
      FileCopy(gvfindtools.z3SrcDir, 'LICENSE.txt', licenseDest),
      MoveFile(licenseDest + os.sep + 'LICENSE.txt', licenseDest + os.sep + 'z3.txt'),
      FileCopy(gvfindtools.z3BinDir, 'z3.exe', gvfindtoolsdeploy.z3BinDir),
    ])
  if args.solver in ['all','cvc4']:
    deployActions.extend([
      FileCopy(gvfindtools.cvc4SrcDir, 'COPYING', licenseDest),
      MoveFile(licenseDest + os.sep + 'COPYING', licenseDest + os.sep + 'cvc4.txt'),
      FileCopy(gvfindtools.cvc4BinDir, 'cvc4.exe', gvfindtoolsdeploy.cvc4BinDir),
    ])

  for action in deployActions:
    action.run()

  logging.info("Deploy finished.")

if __name__ == '__main__':
  main(sys.argv)
