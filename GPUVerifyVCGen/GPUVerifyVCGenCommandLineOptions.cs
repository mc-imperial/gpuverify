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
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;

namespace GPUVerify
{

  public class GPUVerifyVCGenCommandLineOptions
  {
    public static List<string> inputFiles = new List<string>();
    public static string outputFile = null;
    public static bool InvertedTracking = false;
    public static bool DebugGPUVerify = false;
    public static bool OnlyDivergence = false;
    public static bool AdversarialAbstraction = false;
    public static bool EqualityAbstraction = false;
    public static bool Inference = true;
    public static bool ArrayEqualities = false;
    public static bool BarrierAccessChecks = true;
    public static bool ConstantWriteChecks = true;
    public static bool ShowStages = false;
    public static bool ShowUniformityAnalysis = false;
    public static bool DoUniformityAnalysis = true;
    public static bool ShowMayBePowerOfTwoAnalysis = false;
    public static bool ShowArrayControlFlowAnalysis = false;
    public static bool SmartPredication = true;
    public static bool OnlyIntraGroupRaceChecking = false;
    public static bool NoBenign = false;
    public static bool AsymmetricAsserts = false;
    public static bool OnlyLog = false;
    public static bool MathInt = false;
    public static bool AbstractHoudini = false;
    public static bool WarpSync = false;
    public static int WarpSize = 32;
    public static string WarpMethod = "resync";
    public static bool NoWarp = false;
    public static bool OnlyWarp = false;
    public static bool AtomicVsRead = true;
    public static bool AtomicVsWrite = true;
    public static bool RefinedAtomics = true;
    public static bool OptimiseBarrierIntervals = true;
    public static bool EliminateRedundantReadInstrumentation = true;
    public static bool RemovePrivateArrayAccesses = true;
    public static bool NonDeterminiseUninterpretedFunctions = false;
    public static bool IdentifySafeBarriers = true;
    public static bool CheckSingleNonInlinedImpl = false;
    public static bool DoCallSiteAnalysis = false;
    public static bool PruneInfeasibleEdges = true;
    public static List<string> DoNotGenerateCandidates = new List<string>();
    public static List<string> Params = new List<string>();

    public static int Parse(string[] args)
    {
      for (int i = 0; i < args.Length; i++)
      {
        bool hasColonArgument = false;
        string beforeColon;
        string afterColon = null;
        int colonIndex = args[i].IndexOf(':');
        if (colonIndex >= 0 && (args[i].StartsWith("-") || args[i].StartsWith("/"))) {
          hasColonArgument = true;
          beforeColon = args[i].Substring(0, colonIndex);
          afterColon = args[i].Substring(colonIndex + 1);
        } else {
          beforeColon = args[i];
        }

        switch (beforeColon)
        {
          case "-params":
          case "/params":
          if (!hasColonArgument)
          {
            Console.WriteLine("Error: parameter list expected after " + beforeColon + " argument");
            Environment.Exit(1);
          }
          if (!afterColon.StartsWith("[") || !afterColon.EndsWith("]")) 
          {
            Console.WriteLine("Error: parameter list must be enclosed in square brackets.");
            Environment.Exit(1);
          }
          afterColon = afterColon.Trim();
          afterColon = afterColon.Substring(1,afterColon.Length-2);
          Params = new List<string>(afterColon.Split(','));
          if (Params.Count == 0 || Params[0].Length == 0)
          {
            Console.WriteLine("Error: first parameter must be kernel name.");
            Environment.Exit(1);
          }
          break;
          
          case "-invertedTracking":
        case "/invertedTracking":
            InvertedTracking = true;
          break;

          case "-help":
          case "/help":
          case "-?":
          case "/?":
          return -1;

          case "-print":
          case "/print":
          if (!hasColonArgument)
          {
            Console.WriteLine("Error: filename expected after " + beforeColon + " argument");
            Environment.Exit(1);
          }
          Debug.Assert(afterColon != null);
          outputFile = afterColon;
          break;

          case "-debugGPUVerify":
          case "/debugGPUVerify":
          DebugGPUVerify = true;
          break;

          case "-onlyDivergence":
          case "/onlyDivergence":
          OnlyDivergence = true;
          break;

          case "-adversarialAbstraction":
          case "/adversarialAbstraction":
          AdversarialAbstraction = true;
          break;

          case "-equalityAbstraction":
          case "/equalityAbstraction":
          EqualityAbstraction = true;
          break;

          case "-showStages":
          case "/showStages":
          ShowStages = true;
          break;

          case "-noInfer":
          case "/noInfer":
          Inference = false;
          break;

          case "-arrayEqualities":
          case "/arrayEqualities":
          ArrayEqualities = true;
          break;

          case "-showUniformityAnalysis":
          case "/showUniformityAnalysis":
          ShowUniformityAnalysis = true;
          break;

          case "-noUniformityAnalysis":
          case "/noUniformityAnalysis":
          DoUniformityAnalysis = false;
          break;

          case "-showMayBePowerOfTwoAnalysis":
          case "/showMayBePowerOfTwoAnalysis":
          ShowMayBePowerOfTwoAnalysis = true;
          break;

          case "-showArrayControlFlowAnalysis":
          case "/showArrayControlFlowAnalysis":
          ShowArrayControlFlowAnalysis = true;
          break;

          case "-noSmartPredication":
          case "/noSmartPredication":
          SmartPredication = false;
          break;

          case "-onlyIntraGroupRaceChecking":
          case "/onlyIntraGroupRaceChecking":
          OnlyIntraGroupRaceChecking = true;
          break;

          case "-noBenign":
          case "/noBenign":
          NoBenign = true;
          break;

          case "-noBarrierAccessChecks":
          case "/noBarrierAccessChecks":
          BarrierAccessChecks = false;
          break;

          case "-noConstantWriteChecks":
          case "/noConstantWriteChecks":
          ConstantWriteChecks = false;
          break;

          case "-asymmetricAsserts":
          case "/asymmetricAsserts":
          AsymmetricAsserts = true;
          break;

          case "-onlyLog":
          case "/onlyLog":
          OnlyLog = true;
          break;

          case "-mathInt":
          case "/mathInt":
          MathInt = true;
          break;

          case "-abstractHoudini":
          case "/abstractHoudini":
          AbstractHoudini = true;
          break;

          case "-doWarpSync":
          case "/doWarpSync":
          WarpSync = true;
          if (hasColonArgument)
            WarpSize = int.Parse(afterColon);
          break;

          case "-noWarp":
          case "/noWarp":
          WarpMethod = "twopass";
          OnlyWarp = false;
          NoWarp = true;
          break;

          case "-onlyWarp":
          case "/onlyWarp":
          WarpMethod = "twopass";
          OnlyWarp = true;
          NoWarp = false;
          break;

          case "-atomics":
          case "/atomics":
          if (hasColonArgument)
            switch(afterColon)
          {
            case "r":
            AtomicVsRead = true;
            AtomicVsWrite = false;
            break;
            case "w":
            AtomicVsRead = false;
            AtomicVsWrite = true;
            break;
            case "rw":
            AtomicVsRead = true;
            AtomicVsWrite = true;
            break;
            case "none":
            AtomicVsRead = false;
            AtomicVsWrite = false;
            break;
            default:
            AtomicVsRead = true;
            AtomicVsWrite = true;
            break;
          }
          break;

          case "-noRefinedAtomics":
          case "/noRefinedAtomics":
          RefinedAtomics = false;
          break;

          case "-noRemovePrivateArrayAccesses":
          case "/noRemovePrivateArrayAccesses":
          RemovePrivateArrayAccesses = false;
          break;

          case "-noEliminateRedundantReadInstrumentation":
          case "/noEliminateRedundantReadInstrumentation":
          EliminateRedundantReadInstrumentation = false;
          break;

          case "-noOptimiseBarrierIntervals":
          case "/noOptimiseBarrierIntervals":
          OptimiseBarrierIntervals = false;
          break;

          case "-checkSingleNonInlinedImpl":
          case "/checkSingleNonInlinedImpl":
          CheckSingleNonInlinedImpl = true;
          break;

          case "-callSiteAnalysis":
          case "/callSiteAnalysis":
          DoCallSiteAnalysis = true;
          break;

          case "-noCandidate":
          case "/noCandidate":
          DoNotGenerateCandidates.Add(afterColon);
          break;

          case "-noPruneInfeasibleEdges":
          case "/noPruneInfeasibleEdges":
          PruneInfeasibleEdges = false;
          break;

          case "-noSafeBarrierIdentification":
          case "/noSafeBarrierIdentification":
          IdentifySafeBarriers = false;
          break;

          case "-nondeterminiseUninterpretedFunctions":
          case "/nondeterminiseUninterpretedFunctions":
          NonDeterminiseUninterpretedFunctions = true;
          break;

          case "-watchdogRaceChecking":
          case "/watchdogRaceChecking":
          if (!hasColonArgument || (afterColon != "SINGLE" && afterColon != "MULTIPLE"))
          {
            Console.WriteLine("Error: one of 'SINGLE' or 'MULTIPLE' expected after " + beforeColon + " argument");
            Environment.Exit(1);
          }
          if(afterColon == "SINGLE") {
            RaceInstrumentationUtil.RaceCheckingMethod = RaceCheckingMethod.WATCHDOG_SINGLE;
          } else {
            RaceInstrumentationUtil.RaceCheckingMethod = RaceCheckingMethod.WATCHDOG_MULTIPLE;
          }
          break;

          default:
          inputFiles.Add(args[i]);
          break;
        }

      }
      return 0;
    }

    private static bool printedHelp = false;

    public static void Usage()
    {
      // Ensure that we only print the help message once

      if (printedHelp)
      {
        return;
      }
      printedHelp = true;

      Console.WriteLine(@"GPUVerifyVCGen: usage:  GPUVerifyVCGen [ option ... ] [ filename ... ]
  where <option> is one of

  /help                         : this message
  /print:file                   : output bpl file

  /params:[K,v1,...,vn]         : If K is a kernel whose non-array parameters 
                                    are (x1,...,xn), then add the following 
                                    precondition:
                                    __requires(x1==v1 && ... && xn==vn)
                                    An asterisk can be used to denote an 
                                    unconstrained parameter 

  Debugging GPUVerifyVCGen
  ------------------------
  /showArrayControlFlowAnalysis : show results of array control flow analysis
  /showMayBePowerOfTwoAnalysis  : show results of analysis that flags up
                                    variables that might be powers of two
  /showUniformityAnalysis       : show results of uniformity analysis
  /showStages                   : dump intermediate stages of processing to a
                                    a series of files

  Optimisations
  -------------
  /noSmartPredication           : use simple, general predication instead of
                                    default smarter method
  /noUniformityAnalysis         : do not apply uniformity analysis to restrict
                                    predication

  Shared state abstraction
  ------------------------
  /adversarialAbstraction       : completely abstract shared arrays so that
                                    reads are nondeterministic
  /equalityAbstraction          : make shared arrays nondeterministic, but
                                    consistent between threads, at barriers

  Invariant inference
  -------------------
  /noInfer                      : turn off automatic invariant inference
  /noLoopPredicateInvariants    : turn off automatic generation of invariants
                                    about loop predicates.  Occasionally they
                                    can be wrong, hindering verification
  /arrayEqualities              : generate equality candidate invariants for
                                    array variables
  /callSiteAnalysis             : generate procedure preconditions based on
                                    procedure call sites

  Property checking
  -----------------
  /onlyDivergence               : verify freedom from barrier divergence, but
                                    not data races
  /onlyIntraGroupRaceChecking   : do not consider inter-group data races
  /noBenign                     : do not tolerate benign data races
  /noBarrierAccessChecks        : do not check barrier invariant accesses
  /noConstantWriteChecks        : do not check writes to constant space

");
        }
  }
}
