//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify
{
    using System.Collections.Generic;
    using Microsoft.Boogie;

    internal interface IKernelArrayInfo
    {
        IEnumerable<Variable> GetGlobalArrays(bool includeDisabled);

        IEnumerable<Variable> GetGroupSharedArrays(bool includeDisabled);

        IEnumerable<Variable> GetGlobalAndGroupSharedArrays(bool includeDisabled);

        IEnumerable<Variable> GetConstantArrays();

        IEnumerable<Variable> GetPrivateArrays();

        IEnumerable<Variable> GetAllArrays(bool includeDisabled);

        IEnumerable<Variable> GetReadOnlyGlobalAndGroupSharedArrays(bool includeDisabled);

        IEnumerable<Variable> GetAtomicallyAccessedArrays(bool includeDisabled);

        bool ContainsGlobalOrGroupSharedArray(Variable v, bool includeDisabled);

        bool ContainsPrivateArray(Variable v);

        bool ContainsConstantArray(Variable v);

        void AddGlobalArray(Variable v);

        void AddGroupSharedArray(Variable v);

        void AddPrivateArray(Variable v);

        void AddConstantArray(Variable v);

        void AddAtomicallyAccessedArray(Variable v);

        void AddReadOnlyGlobalOrGroupSharedArray(Variable v);

        void RemovePrivateArray(Variable v);

        void DisableGlobalOrGroupSharedArray(Variable v);
    }
}
