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
    public string ConfigFile = null;
    public bool ParallelInference = false;

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

      if (name == "parallelInference") {
        ParallelInference = true;
        return true;
      }

      return base.ParseOption(name, ps);  // defer to superclass
    }
  }
}
