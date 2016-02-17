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

        IEnumerable<Variable> GetGlobalArrays();

        IEnumerable<Variable> GetGroupSharedArrays();

        IEnumerable<Variable> GetGlobalAndGroupSharedArrays();

        IEnumerable<Variable> GetConstantArrays();

        IEnumerable<Variable> GetPrivateArrays();

        IEnumerable<Variable> GetAllArrays();

        IEnumerable<Variable> GetReadOnlyGlobalAndGroupSharedArrays();

        IEnumerable<Variable> GetAtomicallyAccessedArrays();

        bool ContainsGlobalOrGroupSharedArray(Variable v);

        bool ContainsPrivateArray(Variable v);

        bool ContainsConstantArray(Variable v);

        void AddGlobalArray(Variable v);

        void AddGroupSharedArray(Variable v);

        void AddPrivateArray(Variable v);

        void AddConstantArray(Variable v);

        void AddAtomicallyAccessedArray(Variable v);

        void AddReadOnlyGlobalOrGroupSharedArray(Variable v);

        void RemovePrivateArray(Variable v);

    }
}
