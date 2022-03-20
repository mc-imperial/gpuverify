#!/usr/bin/env python3
# vim: set shiftwidth=2 tabstop=2 expandtab softtabstop=2:
import sys
import os
import argparse
import logging
import shutil
import re
import datetime
import subprocess
import fileinput

import getversion

# Try to import the paths need for GPUVerify's tools
GPUVerifyRoot = sys.path[0]

if os.path.isfile(os.path.join(GPUVerifyRoot, 'gvfindtools.py')):
  import gvfindtools
else:
  sys.stderr.write('Error: Cannot find \'gvfindtools.py\'.'
                   'Did you forget to create it from a template?\n')
  sys.exit(1)

# Import the deploy script
sys.path.insert(0, os.path.join(GPUVerifyRoot, 'gvfindtools.templates'))
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
    self.srcpath=os.path.join(srcdir, filename)
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
    return os.path.join(self.destination, self.filename)

# This will copy the contents of srcdir into destdir
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
      fullPath=os.path.join(path, fileOrDirectory)
      if self.copyOnlyRegex.match(fullPath) == None:
        if not os.path.isdir(fullPath):
          #The item is a directory and it didn't match the regex
          #so we add it to the ignore list
          filesToIgnore.append(fileOrDirectory)
          logging.debug('ignoring ' + fullPath)
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

    # compile regex
    logging.info("Searching for files matching regex \"" + self.fileRegex + "\" in \"" + self.srcdir + "\"")
    regex = re.compile(self.fileRegex)

    # loop over files in self.srcdir
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

class StripFile(DeployTask):
  """
      This class is intended to be used to strip binary files.
  """
  def __init__(self,path):
    """
        path  : The full path to the file to strip
    """
    self.path=path

  def run(self):
    if not os.path.isfile(self.path):
      logging.error('File "' + self.path + '" does not exist')

    logging.info('Stripping "' + self.path + '"')
    subprocess.call(['strip', self.path])

class EmbedMonoRuntime(DeployTask):
  def __init__(self, exePath, outputPath, assemblies=None, staticLink=True):
    self.exePath = exePath
    self.outputPath = outputPath
    self.staticLink = staticLink
    self.assemblies = assemblies

    assert os.path.isabs(self.exePath)
    assert not os.path.isdir(self.exePath)
    assert os.path.isabs(self.outputPath)
    assert not os.path.isdir(self.outputPath)

  def run(self):
    workdir = os.path.dirname(self.exePath)
    cmdLine = ['mkbundle',
               '--deps',
               '-o', self.outputPath
              ]

    if self.staticLink:
      cmdLine.append('--static')

    cmdLine.append(self.exePath) # The compiled .NET program that we will create a bundle for

    if self.assemblies != None:
      assert type(self.assemblies) == list
      cmdLine.extend(self.assemblies)

    logging.info('Running {}'.format(cmdLine))
    retCode = subprocess.call(cmdLine)

    if retCode != 0:
      logging.error('Failed to embed mono runtime for {}'.format(self.exePath))
      sys.exit(1)

class RemoveFile(DeployTask):
  def __init__(self, fileToRemove):
    self.fileToRemove = fileToRemove

  def run(self):
    logging.info('Removing file {}'.format(self.fileToRemove))
    if not os.path.isfile(self.fileToRemove):
      logging.error('File "' + self.path + '" does not exist')
      sys.exit(1)

    os.remove(self.fileToRemove)

class InPlaceSubstitution(DeployTask):
  def __init__(self, filePath, subs):
    self.filePath = filePath
    self.subs = subs

    assert os.path.isabs(self.filePath)
    assert type(self.subs) == dict

  def run(self):
    logging.info('Doing substitutions in file "{}" using subs {}'.format(self.filePath, self.subs))
    for line in fileinput.input(self.filePath,inplace=True):
      newLine = line
      for key, replacement in self.subs.items():
        oldLine = newLine
        newLine = line.replace( '@' + key + '@', replacement)

        if oldLine != newLine:
          logging.debug('****Replacement\n"{original}"\nwith\n"{replacement}"\n'.format(original=oldLine.rstrip('\n'), replacement=newLine.rstrip('\n')))

      # Finally write new line value to file
      sys.stdout.write(newLine)


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
  parser.add_argument("-m",
                      "--embed-mono-runtime",
                      help="If enabled embed the mono runtime in the C# tools" if os.name == 'posix' else argparse.SUPPRESS,
                      action = "store_true",
                      default = False
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
  versionString = getversion.getVersionStringFromRepos()
  versionString += "Deployed on " + datetime.datetime.utcnow().ctime() + " (UTC)\n"

  #Specify actions to perform
  deployActions = [
  # tutorial
  FileCopy(os.path.join(GPUVerifyRoot, "Documentation"), "tutorial.rst", deployDir),
  MoveFile(os.path.join(deployDir, 'tutorial.rst'), os.path.join(deployDir, 'TUTORIAL.TXT')),

  # libclc
  DirCopy(gvfindtools.libclcInstallDir, gvfindtoolsdeploy.libclcInstallDir),
  FileCopy(gvfindtools.libclcSrcDir, 'LICENSE.TXT', licenseDest),
  MoveFile(os.path.join(licenseDest, 'LICENSE.TXT'), os.path.join(licenseDest, 'libclc.txt')),
  # bugle
  DirCopy(os.path.join(gvfindtools.bugleSrcDir, 'include-blang'), os.path.join(gvfindtoolsdeploy.bugleSrcDir, 'include-blang')),
  FileCopy(gvfindtools.bugleSrcDir, 'LICENSE.TXT', licenseDest),
  MoveFile(os.path.join(licenseDest, 'LICENSE.TXT'), os.path.join(licenseDest, 'bugle.txt')),
  RegexFileCopy(gvfindtools.bugleBinDir, r'bugle(\.exe)?$', gvfindtoolsdeploy.bugleBinDir),
  RegexFileCopy(gvfindtools.bugleBinDir, r'libbugleInlineCheckPlugin\.(so|dylib)?$', gvfindtoolsdeploy.bugleBinDir),
  # GPUVerify
  FileCopy(GPUVerifyRoot, 'GPUVerify.py', deployDir),
  FileCopy(GPUVerifyRoot, 'getversion.py', deployDir),
  FileCopy(GPUVerifyRoot, 'LICENSE.TXT', licenseDest),
  MoveFile(os.path.join(licenseDest, 'LICENSE.TXT'), os.path.join(licenseDest, 'gpuverify-boogie.txt')),
  IfUsing('posix',FileCopy(GPUVerifyRoot, 'gpuverify', deployDir)),
  IfUsing('nt',FileCopy(GPUVerifyRoot, 'GPUVerify.bat', deployDir)),
  FileCopy(os.path.join(GPUVerifyRoot, 'gvfindtools.templates'), 'gvfindtoolsdeploy.py', deployDir), # Note this will patched later
  MoveFile(os.path.join(deployDir, 'gvfindtoolsdeploy.py'), os.path.join(deployDir, 'gvfindtools.py')),
  RegexFileCopy(gvfindtools.gpuVerifyBinDir, r'^.+\.(dll|exe)$', gvfindtoolsdeploy.gpuVerifyBinDir),
  FileCopy(GPUVerifyRoot, 'gvtester.py', deployDir),
  DirCopy(os.path.join(GPUVerifyRoot ,'testsuite'), os.path.join(deployDir, 'testsuite')),
  DirCopy(os.path.join(GPUVerifyRoot ,'GPUVerifyScript'), os.path.join(deployDir, 'GPUVerifyScript'), copyOnlyRegex=r'^.+\.py$'),
  # llvm, clang
  FileCopy(gvfindtools.llvmSrcDir, 'LICENSE.TXT', licenseDest),
  MoveFile(os.path.join(licenseDest, 'LICENSE.TXT'), os.path.join(licenseDest, 'llvm.txt')),
  FileCopy(os.path.join(gvfindtools.llvmSrcDir, 'tools', 'clang'), 'LICENSE.TXT', licenseDest),
  MoveFile(os.path.join(licenseDest, 'LICENSE.TXT'), os.path.join(licenseDest, 'clang.txt')),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^clang(\.exe)?$', gvfindtoolsdeploy.llvmBinDir ),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^opt(\.exe)?$', gvfindtoolsdeploy.llvmBinDir),
  RegexFileCopy(gvfindtools.llvmBinDir, r'^llvm-nm(\.exe)?$', gvfindtoolsdeploy.llvmBinDir),
  DirCopy(gvfindtools.llvmLibDir, gvfindtoolsdeploy.llvmLibDir, copyOnlyRegex=r'^.+\.h$'), # Only Copy clang header files
  # file for version information
  CreateFileFromString(versionString, os.path.join(deployDir, os.path.basename(getversion.GPUVerifyDeployVersionFile))),
  # license file
  CreateFileFromString(licenseString, os.path.join(deployDir, "LICENSE.TXT"))
  ]

  # solvers
  if args.solver in ['all','z3']:
    deployActions.extend([
      FileCopy(gvfindtools.z3SrcDir, 'LICENSE.txt', licenseDest),
      MoveFile(os.path.join(licenseDest, 'LICENSE.txt'), os.path.join(licenseDest, 'z3.txt')),
      FileCopy(gvfindtools.z3BinDir, 'z3.exe', gvfindtoolsdeploy.z3BinDir),
    ])
  if args.solver in ['all','cvc4']:
    deployActions.extend([
      FileCopy(gvfindtools.cvc4SrcDir, 'COPYING', licenseDest),
      MoveFile(os.path.join(licenseDest, 'COPYING'), os.path.join(licenseDest, 'cvc4.txt')),
      FileCopy(gvfindtools.cvc4BinDir, 'cvc4.exe', gvfindtoolsdeploy.cvc4BinDir),
      IfUsing('posix', StripFile(os.path.join(gvfindtoolsdeploy.cvc4BinDir, 'cvc4.exe')))
    ])

  # Embed mono runtime
  if args.embed_mono_runtime:
    # Make a list of the assemblies that need embedding and the GPUVerify executable names
    _ignored, _ignored, files = next(os.walk(gvfindtools.gpuVerifyBinDir))

    assemblies = list(filter(lambda f: f.endswith('.dll'), files))
    gpuverifyExecutables = list(filter(lambda f: f.startswith('GPUVerify') and f.endswith('.exe'), files))
    assert len(assemblies) > 0
    assert len(gpuverifyExecutables) > 0

    for tool in gpuverifyExecutables:
      deployActions.append(
        EmbedMonoRuntime(exePath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, tool),
                         outputPath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, tool + ".mono"),
                         assemblies = map(lambda a: os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, a), assemblies)))

    # Delete the assemblies after creating the bundled executables
    for fileToRemove in assemblies + gpuverifyExecutables:
      deployActions.append(
          RemoveFile(os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, fileToRemove)))

    # Finally rename the bundled executables (e.g. GPUVerifyVCGen.exe.mono -> GPUVerifyVCGen.exe)
    for executable in gpuverifyExecutables:
      deployActions.append(
        MoveFile(srcpath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, executable + '.mono'),
                 destpath = os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, executable)))
      deployActions.append(StripFile(os.path.join(gvfindtoolsdeploy.gpuVerifyBinDir, executable)))

  substitutions = { 'USE_MONO': str((not args.embed_mono_runtime) and os.name == 'posix') }

  # Write out gvfindtools.py with any necessary substitutions
  deployActions.append(InPlaceSubstitution(filePath = os.path.join(deployDir, 'gvfindtools.py'),
                                           subs = substitutions))

  for action in deployActions:
    action.run()

  logging.info("Deploy finished.")

if __name__ == '__main__':
  main(sys.argv)
