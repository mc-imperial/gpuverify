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
  using System.Text.RegularExpressions;
  using System.Linq;

  /// <summary>
  /// Wrapper class for a concurrent Houdini instance. It is able to run either on
  /// the main thread or on a worker thread.
  /// </summary>
  public class StaticRefutationEngine : RefutationEngine
  {
    private Houdini.ConcurrentHoudini houdini = null;
    private bool isTrusted;
    private string solver;
    private int errorLimit;
    private bool disableLEI;
    private bool disableLMI;
    private bool modifyTSO;
    private int loopUnroll;

    public override int ID { get { return this.id; } }
    public override string Name { get { return this.name; } }
    public bool IsTrusted { get { return this.isTrusted; } }
    public Houdini.ConcurrentHoudini Houdini { get { return this.houdini; } }

    public StaticRefutationEngine(int id, string name, string solver, string errorLimit, string disableLEI,
                                  string disableLMI, string modifyTSO, string loopUnroll)
    {
      this.id = id;
      this.name = name;
      this.solver = solver;
      this.errorLimit = int.Parse(errorLimit);
      this.disableLEI = bool.Parse(disableLEI);
      this.disableLMI = bool.Parse(disableLMI);
      this.modifyTSO = bool.Parse(modifyTSO);
      this.loopUnroll = int.Parse(loopUnroll);

      CommandLineOptions.Clo.Cho.Add(new CommandLineOptions.ConcurrentHoudiniOptions());
      CommandLineOptions.Clo.Cho[id].ProverCCLimit = this.errorLimit;
      CommandLineOptions.Clo.Cho[id].DisableLoopInvEntryAssert = this.disableLEI;
      CommandLineOptions.Clo.Cho[id].DisableLoopInvMaintainedAssert = this.disableLMI;
      CommandLineOptions.Clo.Cho[id].ModifyTopologicalSorting = this.modifyTSO;

      CommandLineOptions.Clo.Cho[id].ProverOptions.AddRange(CommandLineOptions.Clo.ProverOptions);
      if (solver.Equals("cvc4"))
        CommandLineOptions.Clo.Cho[id].ProverOptions.Add("SOLVER=cvc4");

      if (this.disableLEI || this.disableLMI || this.loopUnroll != -1)
        this.isTrusted = false;
      else
        this.isTrusted = true;
    }

    /// <summary>
    /// Starts a new concurrent Houdini execution. Returns the outcome of the
    /// Houdini process by reference.
    /// </summary>
    public int start(Program program, ref Houdini.HoudiniOutcome outcome)
    {
      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] started crunching ...");
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferInfo)
        printConfig();

      if (loopUnroll != -1)
        program.UnrollLoops(loopUnroll, CommandLineOptions.Clo.SoundLoopUnrolling);

      var houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();
      houdini = new Houdini.ConcurrentHoudini(id, program, houdiniStats, "houdiniCexTrace_" + id +".bpl");

      if (outcome != null)
        outcome = houdini.PerformHoudiniInference(initialAssignment: outcome.assignment);
      else
        outcome = houdini.PerformHoudiniInference();

      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] finished.");

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DebugConcurrentHoudini) {
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

      return id;
    }

    /// <summary>
    /// Prints the configuration options of the Static Refutation Engine.
    /// </summary>
    public override void printConfig()
    {
      Console.WriteLine("######################################");
      Console.WriteLine("# Configuration for " + name + ":");
      Console.WriteLine("# id = " + id);
      Console.WriteLine("# solver = " + solver);
      Console.WriteLine("# errorLimit = " + errorLimit);
      Console.WriteLine("# disableLEI = " + disableLEI);
      Console.WriteLine("# disableLMI = " + disableLMI);
      Console.WriteLine("# modifyTSO = " + modifyTSO);
      Console.WriteLine("# loopUnroll = " + loopUnroll);
      Console.WriteLine("######################################");
    }
  }

  /// <summary>
  /// Wrapper class for a concurrent dynamic analyser instance. It is able to run either on
  /// the main thread or on a worker thread.
  /// </summary>
  public class DynamicRefutationEngine : RefutationEngine
  {
    private int threadId_X;
    private int threadId_Y;
    private int threadId_Z;
    private int groupId_X;
    private int groupId_Y;
    private int groupId_Z;

    public override int ID { get { return this.id; } }
    public override string Name { get { return this.name; } }

    public DynamicRefutationEngine(int id, string name, string threadId_X, string threadId_Y, string threadId_Z,
                                   string groupId_X, string groupId_Y, string groupId_Z)
    {
      this.id = id;
      this.name = name;
      this.threadId_X = checkForMaxAndParseInt(threadId_X);
      this.threadId_Y = checkForMaxAndParseInt(threadId_Y);
      this.threadId_Z = checkForMaxAndParseInt(threadId_Z);
      this.groupId_X = checkForMaxAndParseInt(groupId_X);
      this.groupId_Y = checkForMaxAndParseInt(groupId_Y);
      this.groupId_Z = checkForMaxAndParseInt(groupId_Z);
    }

    /// <summary>
    /// Starts a new concurrent dynamic analyser execution.
    /// </summary>
    public void start(Program program, bool verbose = false, int debug = 0)
    {
      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] started crunching ...");
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferInfo)
        printConfig();

      DynamicAnalysis.MainClass.Start(program, 
                                      Tuple.Create(threadId_X, threadId_Y, threadId_Z), 
                                      Tuple.Create(groupId_X, groupId_Y, groupId_Z),
                                      verbose, debug);

      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] finished.");
    }

    private int checkForMaxAndParseInt(string value)
    {
      if (value.ToLower().Equals("max")) return int.MaxValue;
      else return int.Parse(value);
    }

    /// <summary>
    /// Prints the configuration options of the Static Refutation Engine.
    /// </summary>
    public override void printConfig()
    {
      Console.WriteLine("######################################");
      Console.WriteLine("# Configuration for " + name + ":");
      Console.WriteLine("# id = " + id);
      Console.WriteLine("# threadId_X = " + threadId_X);
      Console.WriteLine("# threadId_Y = " + threadId_Y);
      Console.WriteLine("# threadId_Z = " + threadId_Z);
      Console.WriteLine("# groupId_X = " + groupId_X);
      Console.WriteLine("# groupId_Y = " + groupId_Y);
      Console.WriteLine("# groupId_Z = " + groupId_Z);
      Console.WriteLine("######################################");
    }
  }

  /// <summary>
  /// An abstract class for a refutation engine.
  /// </summary>
  public abstract class RefutationEngine
  {
    protected int id;
    protected string name;

    public abstract int ID { get; }
    public abstract string Name { get; }

    /// <summary>
    /// Prints the configuration options of the Refutation Engine.
    /// </summary>
    public abstract void printConfig();
  }
}
