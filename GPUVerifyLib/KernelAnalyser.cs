//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using Microsoft.Boogie;
using VC;

namespace GPUVerify
{
  public static class KernelAnalyser
  {
    public enum PipelineOutcome
    {
      Done,
      ResolutionError,
      TypeCheckingError,
      ResolvedAndTypeChecked,
      FatalError,
      VerificationCompleted
    }

    /// <summary>
    /// Resolves and type checks the given Boogie program.  Any errors are reported to the
    /// console.  Returns:
    ///  - Done if no errors occurred, and command line specified no resolution or no type checking.
    ///  - ResolutionError if a resolution error occurred
    ///  - TypeCheckingError if a type checking error occurred
    ///  - ResolvedAndTypeChecked if both resolution and type checking succeeded
    /// </summary>
    public static PipelineOutcome ResolveAndTypecheck(Program program, string bplFileName)
    {
      Contract.Requires(program != null);
      Contract.Requires(bplFileName != null);

      // ---------- Resolve ----------

      if (CommandLineOptions.Clo.NoResolve) {
        return PipelineOutcome.Done;
      }

      int errorCount = program.Resolve();
      if (errorCount != 0) {
        Console.WriteLine("{0} name resolution errors detected in {1}", errorCount, bplFileName);
        return PipelineOutcome.ResolutionError;
      }

      // ---------- Type check ----------

      if (CommandLineOptions.Clo.NoTypecheck) {
        return PipelineOutcome.Done;
      }

      errorCount = program.Typecheck();
      if (errorCount != 0) {
        Console.WriteLine("{0} type checking errors detected in {1}", errorCount, bplFileName);
        return PipelineOutcome.TypeCheckingError;
      }

      if (CommandLineOptions.Clo.PrintFile != null && CommandLineOptions.Clo.PrintDesugarings) {
        // if PrintDesugaring option is engaged, print the file here, after resolution and type checking
        GVUtil.IO.PrintBplFile(CommandLineOptions.Clo.PrintFile, program, true);
      }

      return PipelineOutcome.ResolvedAndTypeChecked;
    }

    public static void EliminateDeadVariables(Program program)
    {
      Contract.Requires(program != null);

      // Eliminate dead variables
      Microsoft.Boogie.UnusedVarEliminator.Eliminate(program);

      // Collect mod sets
      if (CommandLineOptions.Clo.DoModSetAnalysis) {
        new ModSetCollector().DoModSetAnalysis(program);
      }

      // Coalesce blocks
      if (CommandLineOptions.Clo.CoalesceBlocks) {
        if (CommandLineOptions.Clo.Trace)
          Console.WriteLine("Coalescing blocks...");
        Microsoft.Boogie.BlockCoalescer.CoalesceBlocks(program);
      }
    }

    public static void Inline(Program program)
    {
      Contract.Requires(program != null);

      // Inline
      var TopLevelDeclarations = cce.NonNull(program.TopLevelDeclarations);

      if (CommandLineOptions.Clo.ProcedureInlining != CommandLineOptions.Inlining.None) {
        bool inline = false;
        foreach (var d in TopLevelDeclarations) {
          if (d.FindExprAttribute("inline") != null) {
            inline = true;
          }
        }
        if (inline) {
          foreach (var d in TopLevelDeclarations) {
            var impl = d as Implementation;
            if (impl != null) {
              impl.OriginalBlocks = impl.Blocks;
              impl.OriginalLocVars = impl.LocVars;
            }
          }
          foreach (var d in TopLevelDeclarations) {
            var impl = d as Implementation;
            if (impl != null && !impl.SkipVerification) {
              Inliner.ProcessImplementation(program, impl);
            }
          }
          foreach (var d in TopLevelDeclarations) {
            var impl = d as Implementation;
            if (impl != null) {
              impl.OriginalBlocks = null;
              impl.OriginalLocVars = null;
            }
          }
        }
      }
    }

    public static void DisableRaceChecking(Program program)
    {
      foreach (var block in program.Blocks()) {
        List<Cmd> newCmds = new List<Cmd>();
        foreach (Cmd c in block.Cmds) {
          CallCmd callCmd = c as CallCmd;
          // TODO: refine into proper check
          if(callCmd == null || !(callCmd.callee.Contains("_CHECK_READ") ||
                                  callCmd.callee.Contains("_CHECK_WRITE") ||
                                  callCmd.callee.Contains("_CHECK_ATOMIC"))) {
            newCmds.Add(c);
          }
        }
        block.Cmds = newCmds;
      }
    }

    public static void DisableRaceLogging(Program program)
    {
      foreach (var block in program.Blocks()) {
        List<Cmd> newCmds = new List<Cmd>();
        foreach (Cmd c in block.Cmds) {
          CallCmd callCmd = c as CallCmd;
          // TODO: refine into proper check
          if (callCmd == null || !(callCmd.callee.Contains("_LOG_READ") ||
                                   callCmd.callee.Contains("_LOG_WRITE") ||
                                   callCmd.callee.Contains("_LOG_ATOMIC"))) {
            newCmds.Add(c);
          }
        }
        block.Cmds = newCmds;
      }
    }

    private static bool IsCandidateAssert(AssertCmd c) {
      return QKeyValue.FindStringAttribute(c.Attributes, "tag") != null;
    }

    public static void DisableAssertions(Program program)
    {
      // We want to disable all assertions, with the exception
      // of assetions at loop heads (these are invariants)
      // and candidate assertions that the user has provided
      // for Houdini
      foreach(var impl in program.Implementations) {
        var CFG = Program.GraphFromImpl(impl);
        CFG.ComputeLoops();
        foreach(var b in impl.Blocks) {
          var newCmds = new List<Cmd>();
          bool CmdCouldBeLoopInvariant = CFG.Headers.Contains(b);
          foreach(var c in b.Cmds) {
            if(c is AssertCmd) {
              if(IsCandidateAssert(c as AssertCmd) || CmdCouldBeLoopInvariant) {
                // Keep it: it's a Houdini candidate or a loop invariant
                newCmds.Add(c);
              } else {
                // Discard the invariant (by not adding it the the new list of commands)
              }
            } else {
              newCmds.Add(c);
            }
            if(!(c is AssertCmd || c is AssumeCmd)) {
              // On seeing a command that isn't an assertion or an assumption,
              // we must be past the sequence of assert and assume commands that
              // form invariants if the block in which they live is a loop header.
              // Thus future commands in this block cannot be invariants
              CmdCouldBeLoopInvariant = false;
            }
          }
          b.Cmds = newCmds;
        }
      }
      

    }

    public static void DisableBarrierDivergenceChecking(Program program)
    {
      foreach (var proc in program.TopLevelDeclarations.OfType<Procedure>()) {
        List<Requires> newRequires = new List<Requires>();
        foreach (Requires r in proc.Requires) {
          if (!QKeyValue.FindBoolAttribute(r.Attributes, "barrier_divergence"))
            newRequires.Add(r);
        }
        proc.Requires = newRequires;
      }
    }

    /// <summary>
    /// Checks if Quantifiers exists in the Boogie program. If they exist and the underlying
    /// parser is CVC4 then it enables the corresponding Logic.
    /// </summary>
    public static void CheckForQuantifiersAndSpecifyLogic(Program program, int taskID = -1)
    {
      var ProverOptions = (taskID >= 0 ? CommandLineOptions.Clo.Cho[taskID].ProverOptions
        : CommandLineOptions.Clo.ProverOptions);

      if ((ProverOptions.Contains("SOLVER=cvc4") ||
            ProverOptions.Contains("SOLVER=CVC4")) &&
          ProverOptions.Contains("LOGIC=QF_ALL_SUPPORTED") &&
          CheckForQuantifiers.Found(program)) {
        ProverOptions.Remove("LOGIC=QF_ALL_SUPPORTED");
        ProverOptions.Add("LOGIC=ALL_SUPPORTED");
      }
    }

    public static void ProcessOutcome(Program program, string implName, VC.VCGen.Outcome outcome, List<Counterexample> errors, string timeIndication,
                               ref int errorCount, ref int verified, ref int inconclusives, ref int timeOuts, ref int outOfMemories)
    {
      switch (outcome) {
        default:
        Contract.Assert(false);  // unexpected outcome
        throw new cce.UnreachableException();
        case VCGen.Outcome.ReachedBound:
        GVUtil.IO.Inform(String.Format("{0}verified", timeIndication));
        Console.WriteLine(string.Format("Stratified Inlining: Reached recursion bound of {0}", CommandLineOptions.Clo.RecursionBound));
        verified++;
        break;
        case VCGen.Outcome.Correct:
        if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed) {
          GVUtil.IO.Inform(String.Format("{0}credible", timeIndication));
          verified++;
        }
        else {
          GVUtil.IO.Inform(String.Format("{0}verified", timeIndication));
          verified++;
        }
        break;
        case VCGen.Outcome.TimedOut:
        timeOuts++;
        GVUtil.IO.Inform(String.Format("{0}timed out", timeIndication));
        break;
        case VCGen.Outcome.OutOfMemory:
        outOfMemories++;
        GVUtil.IO.Inform(String.Format("{0}out of memory", timeIndication));
        break;
        case VCGen.Outcome.Inconclusive:
        inconclusives++;
        GVUtil.IO.Inform(String.Format("{0}inconclusive", timeIndication));
        break;
        case VCGen.Outcome.Errors:
        if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed) {
          GVUtil.IO.Inform(String.Format("{0}doomed", timeIndication));
          errorCount++;
        } //else {
        Contract.Assert(errors != null);  // guaranteed by postcondition of VerifyImplementation
        {
          // BP1xxx: Parsing errors
          // BP2xxx: Name resolution errors
          // BP3xxx: Typechecking errors
          // BP4xxx: Abstract interpretation errors (Is there such a thing?)
          // BP5xxx: Verification errors

          errors.Sort(new CounterexampleComparer());

          foreach (Counterexample error in errors)
          {
            new GPUVerifyErrorReporter(program, implName).ReportCounterexample(error);
            errorCount++;
          }
          //}
          GVUtil.IO.Inform(String.Format("{0}error{1}", timeIndication, errors.Count == 1 ? "" : "s"));
        }
        break;
      }
    }
  }
}
