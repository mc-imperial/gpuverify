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
  using VC;

  public class RefutationEngine
  {
    public int id;
    public string name;
    public bool isTrusted;

    private string solver;
    private int errorLimit;
    private bool disableLMI;
    private bool modifyTSO;
    private int loopUnwind;

    public Houdini.ConcurrentHoudini houdini = null;

    public RefutationEngine(int id, string name, string solver, string errorLimit, string disableLMI, string modifyTSO, string loopUnwind)
    {
      this.id = id;
      this.name = name;
      this.solver = solver;
      this.errorLimit = int.Parse(errorLimit);
      this.disableLMI = bool.Parse(disableLMI);
      this.modifyTSO = bool.Parse(modifyTSO);
      this.loopUnwind = int.Parse(loopUnwind);

      CommandLineOptions.Clo.Cho.Add(new CommandLineOptions.ConcurrentHoudiniOptions());
      CommandLineOptions.Clo.Cho[id].LoopUnrollCount = this.errorLimit;
      CommandLineOptions.Clo.Cho[id].DisableLoopInvMaintainedAssert = this.disableLMI;
      CommandLineOptions.Clo.Cho[id].ModifyTopologicalSorting = this.modifyTSO;
      CommandLineOptions.Clo.Cho[id].LoopUnrollCount = this.loopUnwind;
    }

    public int run(Program program)
    {
      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("INFO:[Thread-" + (id+1) + "] running " + name + " refutation engine.");
        printConfig();
      }

      var houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();
      houdini = new Houdini.ConcurrentHoudini(id, program, houdiniStats, "houdiniCexTrace_" + id +".bpl");
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

        Console.WriteLine("INFO:[Thread-" + (id+1) + "] finished computing the assignments");

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

      return 0;
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

    public void printConfig()
    {
      Console.WriteLine("### Configuration for " + name + " ###");
      Console.WriteLine("id = " + id);
      Console.WriteLine("solver = " + solver);
      Console.WriteLine("errorLimit = " + errorLimit);
      Console.WriteLine("disableLMI = " + disableLMI);
      Console.WriteLine("modifyTSO = " + modifyTSO);
      Console.WriteLine("loopUnwind = " + loopUnwind);
    }
  }
}
