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
using Microsoft.Boogie;

namespace GPUVerify
{
    interface IKernelArrayInfo
    {

        ICollection<Variable> GetGlobalArrays();

        ICollection<Variable> GetGroupSharedArrays();

        ICollection<Variable> GetGlobalAndGroupSharedArrays();

        ICollection<Variable> GetConstantArrays();

        ICollection<Variable> GetPrivateArrays();

        ICollection<Variable> GetAllArrays();

        ICollection<Variable> GetReadOnlyGlobalAndGroupSharedArrays();

        ICollection<Variable> GetAtomicallyAccessedArrays();

        bool ContainsGlobalOrGroupSharedArray(Variable v);

        bool ContainsPrivateArray(Variable v);

        bool ContainsConstantArray(Variable v);

    }
}
