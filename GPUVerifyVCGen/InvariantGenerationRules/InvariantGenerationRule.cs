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

namespace GPUVerify.InvariantGenerationRules
{
    abstract class InvariantGenerationRule
    {
        protected GPUVerifier verifier;

        public InvariantGenerationRule(GPUVerifier verifier)
        {
            this.verifier = verifier;
        }

        public abstract void GenerateCandidates(Implementation Impl, IRegion region);
    }

}
