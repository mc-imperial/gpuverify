//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using GPUVerify;

namespace Microsoft.Boogie
{
  using System;
  using System.IO;
  using System.Collections.Generic;
  using System.Diagnostics;
  using System.Diagnostics.Contracts;
  using System.Linq;
  using VC;

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

        const string DUMP_FILE = "__gvdump.txt";

        #region Give generic internal error messsage
        Console.Error.WriteLine("\nGPUVerify: an internal error has occurred, details written to " + DUMP_FILE + ".");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Please consult the troubleshooting guide in the GPUVerify documentation");
        Console.Error.WriteLine("for common problems, and if this does not help, raise an issue via the");
        Console.Error.WriteLine("GPUVerify issue tracker:");
        Console.Error.WriteLine();
        Console.Error.WriteLine("  https://gpuverify.codeplex.com");
        Console.Error.WriteLine();
        #endregion"

        #region Now try to give the user a specific hint if this looks like a common problem
        try {
          throw e;
        } catch(ProverException) {
          Console.Error.WriteLine("Hint: It looks like GPUVerify is having trouble invoking its");
          Console.Error.WriteLine("supporting theorem prover, which by default is Z3.  Have you");
          Console.Error.WriteLine("installed Z3?");
        } catch(Exception) {
          // Nothing to say about this
        }
        #endregion

        #region Write details of the exception to the dump file
        using (TokenTextWriter writer = new TokenTextWriter(DUMP_FILE)) {
          writer.Write("Exception ToString:");
          writer.Write("===================");
          writer.Write(e.ToString());
          writer.Close();
        }
        #endregion

        Environment.Exit(1);
      }
    }

    static int InferInvariantsInFiles(List<string> fileNames)
    {
      Contract.Requires(cce.NonNullElements(fileNames));

      List<string> filesToProcess = new List<string>();
      filesToProcess.Add(fileNames[fileNames.Count - 1]);

      var annotatedFile = Path.GetDirectoryName(filesToProcess[0]) + Path.VolumeSeparatorChar +
        Path.GetFileNameWithoutExtension(filesToProcess[0]);// + ".inv";

      Houdini.Houdini houdini = null;
      InvariantInference invariantInference = new InvariantInference();

      #region Compute invariant without race checking
      {
        if (CommandLineOptions.Clo.Trace) {
          Console.WriteLine("Compute invariant without race checking");
        }

        Program InvariantComputationProgram = GVUtil.IO.ParseBoogieProgram(fileNames, false);
        if (InvariantComputationProgram == null) return 1;

        KernelAnalyser.PipelineOutcome oc = KernelAnalyser.ResolveAndTypecheck(InvariantComputationProgram, filesToProcess[0]);
        if (oc != KernelAnalyser.PipelineOutcome.ResolvedAndTypeChecked) return 1;

        KernelAnalyser.DisableRaceChecking(InvariantComputationProgram);
        KernelAnalyser.EliminateDeadVariablesAndInline(InvariantComputationProgram);
        KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(InvariantComputationProgram);

        // enable parallelism
        //ConfigureInvariantInference();

        var houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();
        houdini = new Houdini.Houdini(InvariantComputationProgram, houdiniStats);
        Houdini.HoudiniOutcome outcome = houdini.PerformHoudiniInference();

        if (CommandLineOptions.Clo.PrintAssignment) {
          Console.WriteLine("Assignment computed by Houdini:");
          foreach (var x in outcome.assignment) {
            Console.WriteLine(x.Key + " = " + x.Value);
          }
        }

        if (CommandLineOptions.Clo.Trace) {
          int numTrueAssigns = 0;
          foreach (var x in outcome.assignment) {
            if (x.Value)
              numTrueAssigns++;
          }

          Console.WriteLine("Number of true assignments = " + numTrueAssigns);
          Console.WriteLine("Number of false assignments = " + (outcome.assignment.Count - numTrueAssigns));
          Console.WriteLine("Prover time = " + houdiniStats.proverTime.ToString("F2"));
          Console.WriteLine("Unsat core prover time = " + houdiniStats.unsatCoreProverTime.ToString("F2"));
          Console.WriteLine("Number of prover queries = " + houdiniStats.numProverQueries);
          Console.WriteLine("Number of unsat core prover queries = " + houdiniStats.numUnsatCoreProverQueries);
          Console.WriteLine("Number of unsat core prunings = " + houdiniStats.numUnsatCorePrunings);
        }

        if (!AllImplementationsValid(outcome)) {
          int verified = 0;
          int errorCount = 0;
          int inconclusives = 0;
          int timeOuts = 0;
          int outOfMemories = 0;

          foreach (Houdini.VCGenOutcome x in outcome.implementationOutcomes.Values) {
            KernelAnalyser.ProcessOutcome(x.outcome, x.errors, "", ref errorCount, ref verified, ref inconclusives, ref timeOuts, ref outOfMemories);
          }

          GVUtil.IO.WriteTrailer(verified, errorCount, inconclusives, timeOuts, outOfMemories);
          return errorCount + inconclusives + timeOuts + outOfMemories;
        }
      }
      #endregion

      #region Apply computed invariants (if any) to the original program
      {
        Program program = GVUtil.IO.ParseBoogieProgram(filesToProcess, false);
        if (program == null) return 1;

        KernelAnalyser.PipelineOutcome oc = KernelAnalyser.ResolveAndTypecheck(program, filesToProcess[0]);
        if (oc != KernelAnalyser.PipelineOutcome.ResolvedAndTypeChecked) return 1;

        KernelAnalyser.EliminateDeadVariablesAndInline(program);

        CommandLineOptions.Clo.PrintUnstructured = 2;

        if (CommandLineOptions.Clo.LoopUnrollCount != -1) {
          Debug.Assert(!CommandLineOptions.Clo.ContractInfer);
          program.UnrollLoops(CommandLineOptions.Clo.LoopUnrollCount, CommandLineOptions.Clo.SoundLoopUnrolling);
          GPUVerifyErrorReporter.FixStateIds(program);
        }

        if(houdini != null) houdini.ApplyAssignment(program);

        if (File.Exists(filesToProcess[0])) File.Delete(filesToProcess[0]);
        GPUVerify.GVUtil.IO.emitProgram(program, annotatedFile);
      }
      #endregion

      return 0;
    }

    private static void ConfigureInvariantInference()
    {
      ConfigurationFileParser cfp = new ConfigurationFileParser();

      if(GetCommandLineOptions().ParallelInference) {
        cfp.enableParsingOfParallelConfigurations();
      }

      cfp.parseFile(GetCommandLineOptions().InvInferConfigFile);
    }

    private static bool AllImplementationsValid(Houdini.HoudiniOutcome outcome)
    {
      foreach (var vcgenOutcome in outcome.implementationOutcomes.Values.Select(i => i.outcome)) {
        if (vcgenOutcome != VCGen.Outcome.Correct) {
          return false;
        }
      }
      return true;
    }

    private static GPUVerifyCruncherCommandLineOptions GetCommandLineOptions()
    {
      return (GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo;
    }
  }
}
