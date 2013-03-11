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
using Microsoft.Boogie;

namespace GPUVerify {
  public class GPUVerifyBoogieDriverCommandLineOptions : CommandLineOptions {

    public bool StagedInference = false;

    public GPUVerifyBoogieDriverCommandLineOptions() :
      base("GPUVerify", "GPUVerify kernel analyser") {
    }

    protected override bool ParseOption(string name, CommandLineOptionEngine.CommandLineParseState ps) {

      if (name == "stagedInference") {
        StagedInference = true;
        return true;
      }

      return base.ParseOption(name, ps);  // defer to superclass
    }

  }
}
