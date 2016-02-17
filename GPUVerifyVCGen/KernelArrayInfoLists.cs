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
using System.Diagnostics;

namespace GPUVerify
{
    class KernelArrayInfoLists : IKernelArrayInfo
    {
        private List<Variable> GlobalVariables;
        private List<Variable> GroupSharedVariables;
        private List<Variable> ConstantVariables;
        private List<Variable> PrivateVariables;
        private List<Variable> ReadOnlyGlobalAndGroupSharedVariables;
        private List<Variable> AtomicallyAccessedVariables;

        public KernelArrayInfoLists()
        {
            GlobalVariables = new List<Variable>();
            GroupSharedVariables = new List<Variable>();
            ConstantVariables = new List<Variable>();
            PrivateVariables = new List<Variable>();
            ReadOnlyGlobalAndGroupSharedVariables = new List<Variable>();
            AtomicallyAccessedVariables = new List<Variable>();
        }

        public ICollection<Variable> GetGlobalArrays()
        {
            return GlobalVariables;
        }

        public ICollection<Variable> GetGroupSharedArrays()
        {
            return GroupSharedVariables;
        }

        public ICollection<Variable> GetConstantArrays()
        {
            return ConstantVariables;
        }

        public ICollection<Variable> GetPrivateArrays()
        {
            return PrivateVariables;
        }

        public ICollection<Variable> GetGlobalAndGroupSharedArrays()
        {
            List<Variable> all = new List<Variable>();
            all.AddRange(GlobalVariables);
            all.AddRange(GroupSharedVariables);
            return all;
        }

        public ICollection<Variable> GetReadOnlyGlobalAndGroupSharedArrays()
        {
            return ReadOnlyGlobalAndGroupSharedVariables;
        }

        public ICollection<Variable> GetAtomicallyAccessedArrays()
        {
          return AtomicallyAccessedVariables;
        }

        public ICollection<Variable> GetAllArrays()
        {
            List<Variable> all = new List<Variable>();
            all.AddRange(GetGlobalAndGroupSharedArrays());
            all.AddRange(GetConstantArrays());
            all.AddRange(PrivateVariables);
            return all;
        }

        public bool ContainsGlobalOrGroupSharedArray(Variable v)
        {
            return GetGlobalAndGroupSharedArrays().Contains(v);
        }

        public bool ContainsConstantArray(Variable v)
        {
            return ConstantVariables.Contains(v);
        }

        public bool ContainsPrivateArray(Variable v)
        {
            return GetPrivateArrays().Contains(v);
        }

    }
}
