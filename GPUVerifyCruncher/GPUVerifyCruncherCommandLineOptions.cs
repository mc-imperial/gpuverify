//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  public class GPUVerifyCruncherCommandLineOptions : GVCommandLineOptions
  {
    public string ConfigFile = "inference.cfg";
    public string ParallelInferenceScheduling = "default";
    public string RefutationEngine = "";
    public int InferenceSliding = 0;
    public int InferenceSlidingLimit = 1;
    public int DynamicErrorLimit = 0;
    public int DelayHoudini = 0;
    public bool ParallelInference = false;
    public bool DynamicAnalysis = false;
    public bool InferInfo = false;
    public int DynamicAnalysisLoopHeaderLimit = 1000;
    public int DynamicAnalysisUnsoundLoopEscaping = 0;
    public bool DynamicAnalysisSoundLoopEscaping = false;
    public bool ReplaceLoopInvariantAssertions = false;
    public bool EnableBarrierDivergenceChecks = false;

    public GPUVerifyCruncherCommandLineOptions() :
      base() { }

    protected override bool ParseOption(string name, CommandLineOptionEngine.CommandLineParseState ps)
    {
      if (name == "invInferConfigFile") {
        if (ps.ConfirmArgumentCount(1)) {
          ConfigFile = ps.args[ps.i];
        }
        return true;
      }

      if (name == "parallelInferenceScheduling") {
        if (ps.ConfirmArgumentCount(1)) {
          ParallelInferenceScheduling = ps.args[ps.i];
        }
        return true;
      }

      if (name == "refutationEngine") {
        if (ps.ConfirmArgumentCount(1)) {
           RefutationEngine = ps.args[ps.i];
        }
        return true;
      }

      if (name == "inferenceSliding") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref InferenceSliding);
        return true;
      }

      if (name == "inferenceSlidingLimit") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref InferenceSlidingLimit);
        return true;
      }

      if (name == "parallelInference") {
        ParallelInference = true;
        return true;
      }

      if (name == "dynamicErrorLimit") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref DynamicErrorLimit);
        return true;
      }

      if (name == "delayHoudini") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref DelayHoudini);
        return true;
      }

      if (name == "dynamicAnalysis") {
        DynamicAnalysis = true;
        return true;
      }
      
      if (name == "dynamicAnalysisSoundLoopEscaping") {
        DynamicAnalysisSoundLoopEscaping = true;
        return true;
      }
      
      if (name == "dynamicAnalysisUnsoundLoopEscaping") {
         if (ps.ConfirmArgumentCount(1))
           ps.GetNumericArgument(ref DynamicAnalysisUnsoundLoopEscaping);
         return true;
      }

      if (name == "dynamicAnalysisLoopHeaderLimit") {
         if (ps.ConfirmArgumentCount(1))
           ps.GetNumericArgument(ref DynamicAnalysisLoopHeaderLimit);
         return true;
      }

      if (name == "inferInfo") {
        InferInfo = true;
        return true;
      }

      if (name == "replaceLoopInvariantAssertions") {
        ReplaceLoopInvariantAssertions = true;
        return true;
      }

      if (name == "enableBarrierDivergenceChecks") {
        EnableBarrierDivergenceChecks = true;
        return true;
      }

      return base.ParseOption(name, ps);  // defer to superclass
    }
  }
}
