//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using Microsoft.Boogie;

namespace GPUVerify
{
  public class GPUVerifyCruncher
  {
    public static void Main(string[] args)
    {
      Contract.Requires(cce.NonNullElements(args));
      CommandLineOptions.Install(new GPUVerifyCruncherCommandLineOptions());

      try {
        CommandLineOptions.Clo.RunningBoogieFromCommandLine = true;

        if (!CommandLineOptions.Clo.Parse(args)) {
          Environment.Exit(1);
        }
        if (CommandLineOptions.Clo.Files.Count == 0) {
          GVUtil.IO.ErrorWriteLine("GPUVerify: error: no input files were specified");
          Environment.Exit(1);
        }
        if (!CommandLineOptions.Clo.DontShowLogo) {
          Console.WriteLine(CommandLineOptions.Clo.Version);
        }

        List<string> fileList = new List<string>();

        foreach (string file in CommandLineOptions.Clo.Files) {
          string extension = Path.GetExtension(file);
          if (extension != null) {
            extension = extension.ToLower();
          }
          fileList.Add(file);
        }

        foreach (string file in fileList) {
          Contract.Assert(file != null);
          string extension = Path.GetExtension(file);

          if (extension != null) {
            extension = extension.ToLower();
          }
          if (extension != ".bpl") {
            GVUtil.IO.ErrorWriteLine("GPUVerify: error: {0} is not a .bpl file", file);
            Environment.Exit(1);
          }
        }

        int exitCode = InferInvariantsInFiles(fileList);
        Environment.Exit(exitCode);
      } catch (Exception e) {
        if(GetCommandLineOptions().DebugGPUVerify) {
          Console.Error.WriteLine("Exception thrown in GPUVerifyCruncher");
          Console.Error.WriteLine(e);
          throw e;
        }

        GVUtil.IO.DumpExceptionInformation(e);

        Environment.Exit(1);
      }
    }

    static int InferInvariantsInFiles(List<string> fileNames)
    {
      Contract.Requires(cce.NonNullElements(fileNames));
      
      //Scheduler scheduler = new Scheduler(fileNames);
      
      
      InvariantInferrer inferrer = new InvariantInferrer();

      int exitCode = inferrer.inferInvariants(fileNames);
      if (exitCode != 0) return exitCode;

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals(""))
      {
        inferrer.applyInvariantsAndEmitProgram();
      }

      return 0;
    }

    private static GPUVerifyCruncherCommandLineOptions GetCommandLineOptions()
    {
      return (GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo;
    }
  }
}
