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
  using System.Threading;
  using System.Threading.Tasks;
  using System.Text.RegularExpressions;
  using System.Linq;
  using VC;

  /*
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

      addConfigurationOptions();

      // Find the static refutation engines
      Dictionary<string, string> staticEngines = config.getRefutationEngines().
        Where(kvp => kvp.Key.Contains("StaticEngine")).
          ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

      // Initialise static refutation engines
      foreach (KeyValuePair<string, string> kvp in staticEngines) {
        refutationEngines.Add(new StaticRefutationEngine(idCounter, kvp.Value,
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

      if (!((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("")) {
        runSingleRefutationEngine();
        return 0;
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
                ((DynamicRefutationEngine) engine).start(getFreshProgram(false, false, false));
              }, tokenSource.Token
              ));
            }
          }

          if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("default")) {
            Task.WaitAll(unsoundTasks.ToArray());
          }
        }

        // Schedule the unsound refutation engines (if any) for execution
        foreach (RefutationEngine engine in refutationEngines) {
          if (engine is StaticRefutationEngine && !((StaticRefutationEngine) engine).IsTrusted) {
            unsoundTasks.Add(Task.Factory.StartNew(
              () => {
              ((StaticRefutationEngine) engine).start(getFreshProgram(false, false, true), ref outcome);
            }, tokenSource.Token
            ));
          }
        }

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("default") ||
            ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInferenceScheduling.Equals("unsound-first")) {
          Task.WaitAll(unsoundTasks.ToArray());
        }

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DelayHoudini > 0)
        {
          Task.WaitAll(unsoundTasks.ToArray(),
            ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DelayHoudini * 1000);
        }

        // Schedule the sound refutation engines for execution
        foreach (RefutationEngine engine in refutationEngines) {
          if (engine is StaticRefutationEngine && ((StaticRefutationEngine) engine).IsTrusted) {
            soundTasks.Add(Task.Factory.StartNew(
              () => {
              engineIdx = ((StaticRefutationEngine) engine).start(getFreshProgram(false, false, true), ref outcome);
            }, tokenSource.Token
            ));
          }
        }

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicErrorLimit > 0)
        {
            int numOfRefuted = Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count;
            bool done = false;

            while (!done) {
                if (refutationEngines.Count >= Environment.ProcessorCount) break;
                Thread.Sleep(((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicErrorLimit * 1000);
                if (Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count > numOfRefuted) {
                    numOfRefuted = Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count;

                    foreach (RefutationEngine engine in refutationEngines)
                    {
                        if (engine is StaticRefutationEngine && ((StaticRefutationEngine)engine).IsTrusted)
                        {
                            CommandLineOptions.Clo.Cho[engine.ID].ProverCCLimit = 20;
                            done = true;
                        }
                    }
                }
            }
        }

        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferenceSliding > 0) {
          int numOfRefuted = Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count;
          int spawnedEngines = 0;
          
          while (true) {
            if (spawnedEngines >= ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferenceSlidingLimit) break;
            Thread.Sleep(((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferenceSliding * 1000);
            if (Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count > numOfRefuted) {
                numOfRefuted = Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count;

                refutationEngines.Add(new StaticRefutationEngine(refutationEngines.Count, "slided_houdini",
                    CommandLineOptions.Clo.ProverCCLimit.ToString(), "false", "false", "false", "-1"));

                soundTasks.Add(Task.Factory.StartNew(
                    () => {
                    engineIdx = ((StaticRefutationEngine) refutationEngines[refutationEngines.Count -1]).
                        start(getFreshProgram(false, false, true), ref outcome);
                    tokenSource.Cancel(false);
                }, tokenSource.Token
                ));
                spawnedEngines++;
            }
          }
        }
        try {
          Task.WaitAny(soundTasks.ToArray(), tokenSource.Token);
          tokenSource.Cancel(false);
        } catch (OperationCanceledException e) {
          // Should not do anything.
        }
      }
      // Sequential invariant inference
      else {
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysis) {
          ((DynamicRefutationEngine) refutationEngines.
           FirstOrDefault( engine => engine is DynamicRefutationEngine )).
            start(getFreshProgram(false, false, false));
        }

        engineIdx = ((StaticRefutationEngine) refutationEngines.
         FirstOrDefault( engine => engine is StaticRefutationEngine )).
          start(getFreshProgram(false, false, true), ref outcome);
      }

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferInfo) {
        // build map from invariant id (_b[0-9]+) to tag variable (e.g., accessBreak)
        var tagMap = new Dictionary<string, string>();
        Program p = getFreshProgram(false, false, false);
        foreach (Block block in p.TopLevelDeclarations.OfType<Implementation>().Select(item => item.Blocks).SelectMany(item => item)) {
          foreach (AssertCmd cmd in block.Cmds.Where(x => x is AssertCmd)) {
            string tag = QKeyValue.FindStringAttribute(cmd.Attributes, "tag");
            if (tag != null) {
              string c;
              ((StaticRefutationEngine) refutationEngines[engineIdx]).Houdini.MatchCandidate(cmd.Expr, out c);
              tagMap[c] = tag;
            }
          }
        }
        printOutcome(outcome, tagMap);
      }

      if (!AllImplementationsValid(outcome)) {
        int verified = 0;
        int errorCount = 0;
        int inconclusives = 0;
        int timeOuts = 0;
        int outOfMemories = 0;

        foreach (var implOutcome in outcome.implementationOutcomes) {
          KernelAnalyser.ProcessOutcome(getFreshProgram(false, false, false), implOutcome.Key, implOutcome.Value.outcome, implOutcome.Value.errors, "", ref errorCount, ref verified, ref inconclusives, ref timeOuts, ref outOfMemories);
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
      var annotatedFile = directoryContainingFiles + Path.DirectorySeparatorChar +
        Path.GetFileNameWithoutExtension(filesToProcess[0]);

      Program program = getFreshProgram(true, true, false);
      CommandLineOptions.Clo.PrintUnstructured = 2;

      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("Applying inferred invariants (if any) to the original program...");
      }

      if (refutationEngines != null && refutationEngines[engineIdx] != null) {
        ((StaticRefutationEngine) refutationEngines[engineIdx]).Houdini.ApplyAssignment(program);
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ReplaceLoopInvariantAssertions)
          replaceLoopInvariantAssertions(program);
      }

      GPUVerify.GVUtil.IO.EmitProgram(program, annotatedFile, "cbpl");
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

    private void runSingleRefutationEngine()
    {
    
      List<string> filesToProcess = new List<string>();
      filesToProcess.Add(fileNames[fileNames.Count - 1]);
      string directoryContainingFiles = Path.GetDirectoryName (filesToProcess [0]);
      if (string.IsNullOrEmpty (directoryContainingFiles))
        directoryContainingFiles = Directory.GetCurrentDirectory ();
      var annotatedFile = directoryContainingFiles + Path.DirectorySeparatorChar +
        Path.GetFileNameWithoutExtension(filesToProcess[0]);
        Console.WriteLine(annotatedFile);
        
      Houdini.HoudiniOutcome outcome = null;
      RefutationEngine engine = null;

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("houdini"))
      {
        engine = new StaticRefutationEngine(0, "houdini",
          CommandLineOptions.Clo.ProverCCLimit.ToString(), "False", "False", "False", "-1");
      }
      else if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("lmi"))
      {
        engine = new StaticRefutationEngine(0, "lmi",
          CommandLineOptions.Clo.ProverCCLimit.ToString(), "False", "True", "False", "-1");
      }
      else if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("lei"))
      {
        engine = new StaticRefutationEngine(0, "lei",
          CommandLineOptions.Clo.ProverCCLimit.ToString(), "True", "False", "False", "-1");
      }
      else if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("lu1"))
      {
        engine = new StaticRefutationEngine(0, "lu",
        CommandLineOptions.Clo.ProverCCLimit.ToString(), "False", "False", "False", "1");
      }
      else if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("lu2"))
      {
        engine = new StaticRefutationEngine(0, "lu",
          CommandLineOptions.Clo.ProverCCLimit.ToString(), "False", "False", "False", "2");
      }
      else if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("dynamic"))
      {
        engine = new DynamicRefutationEngine(0, "dynamic", "-1", "-1", "-1", "-1", "-1", "-1");
      }
       
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).RefutationEngine.Equals("dynamic"))
      {
        DynamicRefutationEngine _engine = engine as DynamicRefutationEngine;
        _engine.start(getFreshProgram(false, false, false));
        int numFalseAssigns = 0;
        using (StreamWriter fs = File.CreateText(annotatedFile + "-killed-da.txt"))
        {
            foreach (string x in _engine.Interpreter.KilledCandidates()) {
                fs.WriteLine("FALSE: " + x);
                numFalseAssigns++;
            }
        }
        //Console.WriteLine("Number of false assignments = " + numFalseAssigns);
      }
      else
      {
        ((StaticRefutationEngine)engine).start(getFreshProgram(false, false, true), ref outcome);
        int numFalseAssigns = 0;
        using (StreamWriter fs = File.CreateText(annotatedFile + "-killed-" + engine.Name + ".txt"))
        {
            foreach (var x in outcome.assignment) {
                if (!x.Value) 
                {
                    fs.WriteLine("FALSE: " + x.Key);
                    numFalseAssigns++;
                }
            }
        }
        Console.WriteLine("Number of false assignments = " + numFalseAssigns);
      }
    }

    private Program getFreshProgram(bool raceCheck, bool divergenceCheck, bool inline)
    {
      divergenceCheck = divergenceCheck ||
                        ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).EnableBarrierDivergenceChecks;

      return GVUtil.GetFreshProgram(fileNames, raceCheck, divergenceCheck, inline);
    }

    private void printOutcome(Houdini.HoudiniOutcome outcome, Dictionary<string, string> tagMap=null)
    {
      int numTrueAssigns = 0;

      Console.WriteLine("Assignment computed by Houdini:");
      foreach (var x in outcome.assignment) {
        if (x.Value) numTrueAssigns++;
        Console.WriteLine(x.Key + " = " + x.Value);
      }

      Console.WriteLine("Number of true assignments = " + numTrueAssigns);
      Console.WriteLine("Number of false assignments = " + (outcome.assignment.Count - numTrueAssigns));

      if (tagMap != null && tagMap.Count() > 0) {
        Console.WriteLine("Invariant generation results:");
        HashSet<string> tags = new HashSet<string>(tagMap.Values);
        Dictionary<string, int> tagTrue = new Dictionary<string, int>();
        Dictionary<string, int> tagCount = new Dictionary<string, int>();
        foreach (var x in tags) {
          tagTrue[x] = 0;
          tagCount[x] = 0;
        }
        foreach (var x in outcome.assignment) {
          if (tagMap.Keys.Contains(x.Key)) {
            string tag = tagMap[x.Key];
            tagCount[tag]++;
            if (x.Value) tagTrue[tag]++;
          }
        }
        Console.WriteLine("  tag, ntrue, nguessed, hitrate");
        foreach (var x in tags.OrderBy(x => x)) {
          Console.WriteLine("  {0}, {1}, {2}, {3}%", x, tagTrue[x], tagCount[x], ((float) tagTrue[x]) / ((float) tagCount[x]) * 100.0);
        }
      }
    }

    // makes sure that configuration options are universally available
    // independently of the backend solver
    private void addConfigurationOptions()
    {
      if (CommandLineOptions.Clo.ProverOptions.Contains("SOLVER=cvc4")) {
        if (CommandLineOptions.Clo.ProverOptions.Contains("OPTIMIZE_FOR_BV=true")) {
          CommandLineOptions.Clo.Z3Options.Add("RELEVANCY=0");
          CommandLineOptions.Clo.Z3Options.Add("SOLVER=true");
        }
      }
      else {
        CommandLineOptions.Clo.ProverOptions.Add("LOGIC=QF_ALL_SUPPORTED");
      }
    }


    /// <summary>
    /// Replace user supplied loop invariants by assumptions.
    /// </summary>
    private void replaceLoopInvariantAssertions(Program program)
    {
      foreach (Block block in program.Blocks()) {
        List<Cmd> newCmds = new List<Cmd>();
        foreach (Cmd cmd in block.Cmds) {
          AssertCmd assertion = cmd as AssertCmd;
          if (assertion != null &&
              QKeyValue.FindBoolAttribute(assertion.Attributes,
                                          "originated_from_invariant")) {
            AssumeCmd assumption = new AssumeCmd(assertion.tok, assertion.Expr,
                                                 assertion.Attributes);
            newCmds.Add(assumption);
          } else {
            newCmds.Add(cmd);
          }
        }
        block.Cmds = newCmds;
      }
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
  */
}
