//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

﻿using GPUVerify;
using DynamicAnalysis;

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

  /// <summary>
  /// Scheduler for infering invariants using Houdini and/or through dynamic analysis.
  /// It allows for either sequential or concurrent execution of refutation engines
  /// using the Task Parallel Library. Has support for multiple scheduling strategies.
  /// </summary>
  public class InvariantInferrer
  {
    private StaticRefutationEngine[] staticEngines = null;
    private DynamicRefutationEngine[] dynamicEngines = null;
    private Configuration config = null;
    private int engineIdx;
    private List<string> fileNames;

    public InvariantInferrer()
    {
      this.config = new Configuration();
      this.staticEngines = new StaticRefutationEngine[config.getNumberOfEngines("static")];
      this.dynamicEngines = new DynamicRefutationEngine[config.getNumberOfEngines("dynamic")];
      this.engineIdx = 0;
      string conf;

      // Initialise static refutation engines
      for (int i = 0; i < staticEngines.Length; i++) {
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
          conf = config.getValue("ParallelInference", "StaticEngine_" + (i + 1));
        } else {
          conf = config.getValue("Inference", "StaticEngine");
        }

        staticEngines[i] = new StaticRefutationEngine(i, conf,
                                                      config.getValue(conf, "Solver"),
                                                      config.getValue(conf, "ErrorLimit"),
                                                      config.getValue(conf, "DisableLEI"),
                                                      config.getValue(conf, "DisableLMI"),
                                                      config.getValue(conf, "ModifyTSO"),
                                                      config.getValue(conf, "LoopUnroll"));
      }

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysis) {
        // Initialise dynamic refutation engines
        for (int i = 0; i < dynamicEngines.Length; i++) {
          if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
            conf = config.getValue("ParallelInference", "DynamicEngine_" + (i + 1));
          } else {
            conf = config.getValue("Inference", "DynamicEngine");
          }

          dynamicEngines[i] = new DynamicRefutationEngine(i, conf,
                                                          config.getValue(conf, "ThreadID_X"),
                                                          config.getValue(conf, "ThreadID_Y"),
                                                          config.getValue(conf, "ThreadID_Z"),
                                                          config.getValue(conf, "GroupID_X"),
                                                          config.getValue(conf, "GroupID_Y"),
                                                          config.getValue(conf, "GroupID_Z"));
        }
      }
    }

    /// <summary>
    /// Schedules refutation engines for sequential or concurrent execution.
    /// </summary>
    public int inferInvariants(List<string> fileNames)
    {   
      Houdini.HoudiniOutcome outcome = null;
      this.fileNames = fileNames;

      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("Computing invariants without race checking...");
      }

      // Concurrent invariant inference
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
        List<Task> unsoundTasks = new List<Task>();
        List<Task> soundTasks = new List<Task>();
        CancellationTokenSource tokenSource = new CancellationTokenSource();

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysis) {
          // Schedule the dynamic analysis engines (if any) for execution
          foreach (DynamicRefutationEngine dynamicEngine in dynamicEngines) {
            unsoundTasks.Add(Task.Factory.StartNew(
              () => {
              dynamicEngine.start(getFreshProgram(false, false));
            }, tokenSource.Token
            ));
          }

          if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("phased") ||
              ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("dynamic-first")) {
            Task.WaitAll(unsoundTasks.ToArray());
          }
        }

        // Schedule the unsound refutation engines (if any) for execution
        foreach (StaticRefutationEngine staticEngine in staticEngines) {
          if (!staticEngine.IsTrusted) {
            unsoundTasks.Add(Task.Factory.StartNew(
              () => {
              staticEngine.start(getFreshProgram(false, true), ref outcome);
            }, tokenSource.Token
            ));
          }
        }

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("phased") ||
            ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("unsound-first")) {
          Task.WaitAll(unsoundTasks.ToArray());
        }

        // Schedule the sound refutation engines for execution
        foreach (StaticRefutationEngine staticEngine in staticEngines) {
          if (staticEngine.IsTrusted) {
            soundTasks.Add(Task.Factory.StartNew(
              () => {
              engineIdx = staticEngine.start(getFreshProgram(false, true), ref outcome);
            }, tokenSource.Token
            ));
          }
        }
        Task.WaitAny(soundTasks.ToArray());
        tokenSource.Cancel();
      }
      // Sequential invariant inference
      else {
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysis)
          dynamicEngines[0].start(getFreshProgram(false, false), true);
        staticEngines[0].start(getFreshProgram(false, true), ref outcome);
      }

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferInfo)
        printOutcome(outcome);

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

    /// <summary>
    /// Applies computed invariants (if any) to the original program and then emits
    /// the program as a bpl file.
    /// </summary>
    public void applyInvariantsAndEmitProgram()
    {
      List<string> filesToProcess = new List<string>();
      filesToProcess.Add(fileNames[fileNames.Count - 1]);
      string directoryContainingFiles = Path.GetDirectoryName (filesToProcess [0]);
      if (string.IsNullOrEmpty (directoryContainingFiles))
        directoryContainingFiles = Directory.GetCurrentDirectory ();
      var annotatedFile = directoryContainingFiles + Path.VolumeSeparatorChar +
        Path.GetFileNameWithoutExtension(filesToProcess[0]);

      Program program = getFreshProgram(true, true);
      CommandLineOptions.Clo.PrintUnstructured = 2;

      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("Applying inferred invariants (if any) to the original program...");
      }

      if (staticEngines != null && staticEngines[engineIdx] != null) {
        staticEngines[engineIdx].Houdini.ApplyAssignment(program);
      }

      if (File.Exists(filesToProcess[0])) File.Delete(filesToProcess[0]);
      GPUVerify.GVUtil.IO.EmitProgram(program, annotatedFile);
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

    private Program getFreshProgram(bool raceCheck, bool inline)
    {
      KernelAnalyser.PipelineOutcome oc;
      List<string> filesToProcess = new List<string>();
      filesToProcess.Add(fileNames[fileNames.Count - 1]);

      Program program = GVUtil.IO.ParseBoogieProgram(fileNames, false);
      if (program == null) Environment.Exit(1);
      oc = KernelAnalyser.ResolveAndTypecheck(program, filesToProcess[0]);
      if (oc != KernelAnalyser.PipelineOutcome.ResolvedAndTypeChecked) Environment.Exit(1);

      if (!raceCheck) KernelAnalyser.DisableRaceChecking(program);
      KernelAnalyser.EliminateDeadVariables(program);
      if (inline) KernelAnalyser.Inline(program);
      KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(program);

      return program;
    }

    private void printOutcome(Houdini.HoudiniOutcome outcome)
    {
      int numTrueAssigns = 0;

      Console.WriteLine("Assignment computed by Houdini:");
      foreach (var x in outcome.assignment) {
        if (x.Value) numTrueAssigns++;
        Console.WriteLine(x.Key + " = " + x.Value);
      }

      Console.WriteLine("Number of true assignments = " + numTrueAssigns);
      Console.WriteLine("Number of false assignments = " + (outcome.assignment.Count - numTrueAssigns));
    }

    /// <summary>
    /// Configuration for sequential and parallel inference.
    /// </summary>
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

      public int getNumberOfEngines(string type)
      {
        int num = 0;

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
          foreach (var key in info["ParallelInference"].Keys) {
            if (type.ToLower().Equals("static") && key.Contains("StaticEngine"))
              num++;
            else if (type.ToLower().Equals("dynamic") && key.Contains("DynamicEngine"))
              num++;
          }
        } else {
          num++;
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

      /// <summary>
      /// Prints all invariant inference configuration options.
      /// </summary>
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
