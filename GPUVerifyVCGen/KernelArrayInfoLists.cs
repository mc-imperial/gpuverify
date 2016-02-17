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

        public IEnumerable<Variable> GetGlobalArrays()
        {
            return GlobalVariables;
        }

        public IEnumerable<Variable> GetGroupSharedArrays()
        {
            return GroupSharedVariables;
        }

        public IEnumerable<Variable> GetConstantArrays()
        {
            return ConstantVariables;
        }

        public IEnumerable<Variable> GetPrivateArrays()
        {
            return PrivateVariables;
        }

        public IEnumerable<Variable> GetGlobalAndGroupSharedArrays()
        {
            List<Variable> all = new List<Variable>();
            all.AddRange(GlobalVariables);
            all.AddRange(GroupSharedVariables);
            return all;
        }

        public IEnumerable<Variable> GetReadOnlyGlobalAndGroupSharedArrays()
        {
            return ReadOnlyGlobalAndGroupSharedVariables;
        }

        public IEnumerable<Variable> GetAtomicallyAccessedArrays()
        {
          return AtomicallyAccessedVariables;
        }

        public IEnumerable<Variable> GetAllArrays()
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

        public void AddGlobalArray(Variable v) {
          GlobalVariables.Add(v);
        }

        public void AddGroupSharedArray(Variable v) {
          GroupSharedVariables.Add(v);
        }

        public void AddPrivateArray(Variable v) {
          PrivateVariables.Add(v);
        }

        public void AddConstantArray(Variable v) {
          ConstantVariables.Add(v);
        }

        public void AddAtomicallyAccessedArray(Variable v) {
          AtomicallyAccessedVariables.Add(v);
        }

        public void AddReadOnlyGlobalOrGroupSharedArray(Variable v) {
          Debug.Assert(GlobalVariables.Contains(v) || GroupSharedVariables.Contains(v));
          ReadOnlyGlobalAndGroupSharedVariables.Add(v);
        }

        public void RemovePrivateArray(Variable v) {
          Debug.Assert(PrivateVariables.Contains(v));
          PrivateVariables.Remove(v);
        }

    }
}
