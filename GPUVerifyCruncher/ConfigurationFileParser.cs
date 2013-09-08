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
  public class ConfigurationFileParser
  {
    bool parseParallelConfigs = false;

    public ConfigurationFileParser() { }

    public void parseFile(string fileName)
    {

    }

    public void enableParsingOfParallelConfigurations()
    {
      parseParallelConfigs = true;
    }

    public void disableParsingOfParallelConfigurations()
    {
      parseParallelConfigs = false;
    }
  }
}
