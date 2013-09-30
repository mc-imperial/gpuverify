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
  ﻿using System;
  using System.IO;
  using System.Collections.Generic;
  using System.Threading;
  using System.Threading.Tasks;
  using System.Text.RegularExpressions;
  using System.Linq;
  using VC;

  public class InvariantInferrer
  {
    private RefutationEngine[] refutationEngines = null;
    private Configuration config = null;
    private int engineIdx;
    private List<string> fileNames;

    public InvariantInferrer(List<string> fileNames)
    {
      this.config = new Configuration();
      this.refutationEngines = new RefutationEngine[config.getNumberOfEngines()];
      this.engineIdx = 0;
      this.fileNames = fileNames;
      string conf;

      // Initialise refutation engines
      for (int i = 0; i < refutationEngines.Length; i++) {
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
          conf = config.getValue("ParallelInference", "Engine_" + (i + 1));
        } else {
          conf = config.getValue("Inference", "Engine");
        }

        refutationEngines[i] = new RefutationEngine(i, conf,
                                                    config.getValue(conf, "Solver"),
                                                    config.getValue(conf, "ErrorLimit"),
                                                    config.getValue(conf, "DisableLMI"),
                                                    config.getValue(conf, "ModifyTSO"),
                                                    config.getValue(conf, "LoopUnroll"));
      }
    }

    public int inferInvariants(Program program)
    {
      Houdini.HoudiniOutcome outcome = null;

      // Schedules refutation engines for execution
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
        List<Task> unsoundTasks = new List<Task>();
        List<Task> soundTasks = new List<Task>();
        CancellationTokenSource tokenSource = new CancellationTokenSource();

        // Schedule the unsound refutatiom engines for execution
        foreach (RefutationEngine engine in refutationEngines) {
          if (!engine.isTrusted) {
            unsoundTasks.Add(Task.Factory.StartNew(
              () => {
              engine.run(getFreshProgram(), ref outcome);
            }, tokenSource.Token
            ));
          }
        }

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("unsound-first"))
          Task.WaitAll(unsoundTasks.ToArray());

        // Schedule the sound refutation engines for execution
        foreach (RefutationEngine engine in refutationEngines) {
          if (engine.isTrusted) {
            soundTasks.Add(Task.Factory.StartNew(
              () => {
              engineIdx = engine.run(getFreshProgram(), ref outcome);
            }, tokenSource.Token
            ));
          }
        }
        Task.WaitAny(soundTasks.ToArray());
        tokenSource.Cancel();
      } else {
        refutationEngines[0].run(program, ref outcome);
      }

      PrintOutcome(outcome);

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

      return 0;
    }

    public void applyInvariants(Program program)
    {
      if (refutationEngines != null && refutationEngines[engineIdx] != null) {
        refutationEngines[engineIdx].houdini.ApplyAssignment(program);
      }
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

    private Program getFreshProgram()
    {
      KernelAnalyser.PipelineOutcome oc;
      List<string> filesToProcess = new List<string>();
      filesToProcess.Add(fileNames[fileNames.Count - 1]);

      Program program = GVUtil.IO.ParseBoogieProgram(fileNames, false);
      if (program == null) Environment.Exit(1);
      oc = KernelAnalyser.ResolveAndTypecheck(program, filesToProcess[0]);
      if (oc != KernelAnalyser.PipelineOutcome.ResolvedAndTypeChecked) Environment.Exit(1);

      KernelAnalyser.DisableRaceChecking(program);
      KernelAnalyser.EliminateDeadVariablesAndInline(program);
      KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(program);

      return program;
    }

    public static void PrintOutcome(Houdini.HoudiniOutcome outcome, Houdini.HoudiniSession.HoudiniStatistics houdiniStats = null)
    {
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

        if (houdiniStats != null) {
          Console.WriteLine("Prover time = " + houdiniStats.proverTime.ToString("F2"));
          Console.WriteLine("Unsat core prover time = " + houdiniStats.unsatCoreProverTime.ToString("F2"));
          Console.WriteLine("Number of prover queries = " + houdiniStats.numProverQueries);
          Console.WriteLine("Number of unsat core prover queries = " + houdiniStats.numUnsatCoreProverQueries);
          Console.WriteLine("Number of unsat core prunings = " + houdiniStats.numUnsatCorePrunings);
        }
      }
    }

    private class Configuration
    {
      private Dictionary<string, Dictionary<string, string>> info = null;

      public Configuration()
      {
        info = new Dictionary<string, Dictionary<string, string>>();
        updateFromConfigurationFile();
      }

      public string getValue(string key1, string key2)
      {
        return info[key1][key2];
      }

      public int getNumberOfEngines()
      {
        int num = 1;

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
          num = ((Dictionary<string, string>)info ["ParallelInference"]).Count;
        }

        return num;
      }

      private void updateFromConfigurationFile()
      {
        string file = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ConfigFile;

        try {
          using (var fileStream = new FileStream(file, FileMode.Open, FileAccess.Read))
            using (var input = new StreamReader(fileStream)) {
            string entry;
            string key = "";

            while ((entry = input.ReadLine()) != null) {
              entry = Regex.Replace(entry, ";.*", "");
              if (entry.Length == 0) continue;
              if (entry.StartsWith("[")) {
                key = Regex.Replace(entry, "[[\\]]+", "");
                info.Add(key, new Dictionary<string, string>());
              }
              else {
                if (key.Length == 0) throw new Exception();
                string[] tokens = new Regex("[ =\t]+").Split(entry);
                if (tokens.Length != 2) throw new Exception();
                info[key].Add(tokens[0], tokens[1]);
              }
            }
          }
        } catch (FileNotFoundException e) {
          Console.Error.WriteLine("{0}: The configuration file {1} was not found", e.GetType(), file);
          Environment.Exit(1);
        } catch (Exception e) {
          Console.Error.WriteLine("{0}: The file {1} is not properly formatted", e.GetType(), file);
          Environment.Exit(1);
        }
      }

      public void print()
      {
        Console.WriteLine("################################################");
        Console.WriteLine("# Configuration Options for Invariant Inference:");
        info.SelectMany(option => option.Value.Select(opt => "# " + option.Key + " :: " + opt.Key + " :: " + opt.Value))
          .ToList().ForEach(Console.WriteLine);
        Console.WriteLine("################################################");
      }
    }
  }
}
