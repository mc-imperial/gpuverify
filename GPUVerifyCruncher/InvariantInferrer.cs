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
    private List<RefutationEngine> refutationEngines = null;
    private Configuration config = null;
    private int engineIdx;
    private List<string> fileNames;

    public InvariantInferrer()
    {
      this.config = new Configuration();
      this.refutationEngines = new List<RefutationEngine>();
      this.engineIdx = 0;
      int idCounter = 0;

      // Find the static refutation engines
      Dictionary<string, string> staticEngines = config.getRefutationEngines().
        Where(kvp => kvp.Key.Contains("StaticEngine")).
          ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

      // Initialise static refutation engines
      foreach (KeyValuePair<string, string> kvp in staticEngines) {
        refutationEngines.Add(new StaticRefutationEngine(idCounter, kvp.Value,
                                                         config.getValue(kvp.Value, "Solver"),
                                                         config.getValue(kvp.Value, "ErrorLimit"),
                                                         config.getValue(kvp.Value, "DisableLEI"),
                                                         config.getValue(kvp.Value, "DisableLMI"),
                                                         config.getValue(kvp.Value, "ModifyTSO"),
                                                         config.getValue(kvp.Value, "LoopUnroll")));
        idCounter++;
      }

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysis) {
        // Find the dynamic refutation engines (if any)
        Dictionary<string, string> dynamicEngines = config.getRefutationEngines().
          Where(kvp => kvp.Key.Contains("DynamicEngine")).
            ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        // Initialise dynamic refutation engines (if any)
        foreach (KeyValuePair<string, string> kvp in dynamicEngines) {
          refutationEngines.Add(new DynamicRefutationEngine(idCounter, kvp.Value,
                                                            config.getValue(kvp.Value, "ThreadID_X"),
                                                            config.getValue(kvp.Value, "ThreadID_Y"),
                                                            config.getValue(kvp.Value, "ThreadID_Z"),
                                                            config.getValue(kvp.Value, "GroupID_X"),
                                                            config.getValue(kvp.Value, "GroupID_Y"),
                                                            config.getValue(kvp.Value, "GroupID_Z")));
          idCounter++;
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
          foreach (RefutationEngine engine in refutationEngines) {
            if (engine is DynamicRefutationEngine) {
              unsoundTasks.Add(Task.Factory.StartNew(
                () => {
                ((DynamicRefutationEngine) engine).start(getFreshProgram(false, false));
              }, tokenSource.Token
              ));
            }
          }

          if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("phased") ||
              ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("dynamic-first")) {
            Task.WaitAll(unsoundTasks.ToArray());
          }
        }

        // Schedule the unsound refutation engines (if any) for execution
        foreach (RefutationEngine engine in refutationEngines) {
          if (engine is StaticRefutationEngine && !((StaticRefutationEngine) engine).IsTrusted) {
            unsoundTasks.Add(Task.Factory.StartNew(
              () => {
              ((StaticRefutationEngine) engine).start(getFreshProgram(false, true), ref outcome);
            }, tokenSource.Token
            ));
          }
        }

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("phased") ||
            ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("unsound-first")) {
          Task.WaitAll(unsoundTasks.ToArray());
        }

        // Schedule the sound refutation engines for execution
        foreach (RefutationEngine engine in refutationEngines) {
          if (engine is StaticRefutationEngine && ((StaticRefutationEngine) engine).IsTrusted) {
            soundTasks.Add(Task.Factory.StartNew(
              () => {
              engineIdx = ((StaticRefutationEngine) engine).start(getFreshProgram(false, true), ref outcome);
            }, tokenSource.Token
            ));
          }
        }
        Task.WaitAny(soundTasks.ToArray());
        tokenSource.Cancel();
      }
      // Sequential invariant inference
      else {
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysis) {
          ((DynamicRefutationEngine) refutationEngines.
           FirstOrDefault( engine => engine is DynamicRefutationEngine )).
            start(getFreshProgram(false, false), true);
        }

        engineIdx = ((StaticRefutationEngine) refutationEngines.
         FirstOrDefault( engine => engine is StaticRefutationEngine )).
          start(getFreshProgram(false, true), ref outcome);
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

      if (refutationEngines != null && refutationEngines[engineIdx] != null) {
        ((StaticRefutationEngine) refutationEngines[engineIdx]).Houdini.ApplyAssignment(program);
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

      public Dictionary<string, string> getRefutationEngines()
      {
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
          return info["ParallelInference"];
        } else {
          return info["Inference"];
        }
      }

      public string getValue(string key1, string key2)
      {
        string value = "";

        if (info[key1].ContainsKey(key2)) {
          value = info[key1][key2];
        } else {
          switch (key2) {
          case "Solver":
            value = "option-defined";
            break;
          case "ErrorLimit":
            value = "20";
            break;
          case "DisableLEI":
            value = "False";
            break;
          case "DisableLMI":
            value = "False";
            break;
          case "ModifyTSO":
            value = "False";
            break;
          case "LoopUnroll":
            value = "-1";
            break;
          }
        }

        return value;
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
