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

  public class RefutationEngine
  {
    public int id;
    public string name;
    public bool isTrusted;

    private string solver;
    private int errorLimit;
    private bool disableLMI;
    private bool modifyTSO;
    private int loopUnroll;

    public Houdini.ConcurrentHoudini houdini = null;

    public RefutationEngine(int id, string name, string solver, string errorLimit, string disableLMI, string modifyTSO, string loopUnroll)
    {
      this.id = id;
      this.name = name;
      this.solver = solver;
      this.errorLimit = int.Parse(errorLimit);
      this.disableLMI = bool.Parse(disableLMI);
      this.modifyTSO = bool.Parse(modifyTSO);
      this.loopUnroll = int.Parse(loopUnroll);

      CommandLineOptions.Clo.Cho.Add(new CommandLineOptions.ConcurrentHoudiniOptions());
      CommandLineOptions.Clo.Cho[id].ProverCCLimit = this.errorLimit;
      CommandLineOptions.Clo.Cho[id].DisableLoopInvMaintainedAssert = this.disableLMI;
      CommandLineOptions.Clo.Cho[id].ModifyTopologicalSorting = this.modifyTSO;

      if (this.disableLMI || this.loopUnroll != -1)
        this.isTrusted = false;
      else
        this.isTrusted = true;
    }

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

    public void printConfig()
    {
      Console.WriteLine("######################################");
      Console.WriteLine("# Configuration for " + name + ":");
      Console.WriteLine("# id = " + id);
      Console.WriteLine("# solver = " + solver);
      Console.WriteLine("# errorLimit = " + errorLimit);
      Console.WriteLine("# disableLMI = " + disableLMI);
      Console.WriteLine("# modifyTSO = " + modifyTSO);
      Console.WriteLine("# loopUnroll = " + loopUnroll);
      Console.WriteLine("######################################");
    }
  }
}
