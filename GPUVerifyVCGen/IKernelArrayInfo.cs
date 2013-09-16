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

namespace GPUVerify
{
    interface IKernelArrayInfo
    {

        ICollection<Variable> getGlobalArrays();

        ICollection<Variable> getGroupSharedArrays();

        ICollection<Variable> getConstantArrays();

        ICollection<Variable> getPrivateArrays();

        ICollection<Variable> getAllNonLocalArrays();

        ICollection<Variable> getAllArrays();

        bool ContainsNonLocalArray(Variable v);

    }
}
