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
    public bool ParallelInference = false;
    public bool DynamicAnalysis = false;
    public bool InferInfo = false;
    public int DynamicAnalysisHeaderExecutionCount = 1000;
    public bool DynamicAnalysisUnrollLoops = false;

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

      if (name == "parallelInference") {
        ParallelInference = true;
        return true;
      }

      if (name == "dynamicAnalysis") {
        DynamicAnalysis = true;
        return true;
      }

      if (name == "dynamicAnalysisHeaderCount") {
         if (ps.ConfirmArgumentCount(1))
           ps.GetNumericArgument(ref DynamicAnalysisHeaderExecutionCount);
         return true;
      }

      if (name == "dynamicAnalysisUnrollLoops") {
         DynamicAnalysisUnrollLoops = true;
         return true;
      }

      if (name == "inferInfo") {
        InferInfo = true;
        return true;
      }

      return base.ParseOption(name, ps);  // defer to superclass
    }
  }
}
