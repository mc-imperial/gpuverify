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
  public class RefutationEngine
  {
    public int id;
    public string name;
    public bool isTrusted;

    private string solver;
    private int errorLimit;
    private bool disableLEI;
    private bool disableLMI;
    private bool modifyTSO;
    private int loopUnroll;

    public Houdini.ConcurrentHoudini houdini = null;

    public RefutationEngine(int id, string name, string solver, string errorLimit, string disableLEI,
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

      if (name.Equals ("cvc4"))
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
    public int run(Program program, ref Houdini.HoudiniOutcome outcome)
    {
      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("INFO:[Engine-" + name + "] started crunching ...");
        printConfig();
      }

      if (loopUnroll != -1)
        program.UnrollLoops(loopUnroll, CommandLineOptions.Clo.SoundLoopUnrolling);

      var houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();
      houdini = new Houdini.ConcurrentHoudini(id, program, houdiniStats, "houdiniCexTrace_" + id +".bpl");

      if (outcome != null)
        outcome = houdini.PerformHoudiniInference(initialAssignment: outcome.assignment);
      else
        outcome = houdini.PerformHoudiniInference();

      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("INFO:[Engine-" + name + "] finished.");
      }

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DebugParallelHoudini) {
        InvariantInferrer.PrintOutcome(outcome, houdiniStats);
      }

      return id;
    }

    /// <summary>
    /// Prints the configuration options of the Refutation Engine.
    /// </summary>
    public void printConfig()
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
}
