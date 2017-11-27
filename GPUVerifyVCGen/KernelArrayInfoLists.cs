//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System.Collections.Generic;
using System.Linq;
using Microsoft.Boogie;
using System.Diagnostics;

namespace GPUVerify
{
    class KernelArrayInfoLists : IKernelArrayInfo
    {
        private List<Variable> GlobalArrays;
        private List<Variable> GroupSharedArrays;
        private List<Variable> ConstantArrays;
        private List<Variable> PrivateArrays;
        private List<Variable> ReadOnlyGlobalAndGroupSharedArrays;
        private List<Variable> AtomicallyAccessedArrays;
        private List<Variable> DisabledArrays;

        public KernelArrayInfoLists()
        {
            GlobalArrays = new List<Variable>();
            GroupSharedArrays = new List<Variable>();
            ConstantArrays = new List<Variable>();
            PrivateArrays = new List<Variable>();
            ReadOnlyGlobalAndGroupSharedArrays = new List<Variable>();
            AtomicallyAccessedArrays = new List<Variable>();
            DisabledArrays = new List<Variable>();
        }

        private IEnumerable<Variable> FilterDisabled(IEnumerable<Variable> arrays, bool includeDisabled) {
          return arrays.Where(Item => includeDisabled || !DisabledArrays.Contains(Item));
        }

        public IEnumerable<Variable> GetGlobalArrays(bool includeDisabled)
        {
            return FilterDisabled(GlobalArrays, includeDisabled);
        }

        public IEnumerable<Variable> GetGroupSharedArrays(bool includeDisabled)
        {
            return FilterDisabled(GroupSharedArrays, includeDisabled);
        }

        public IEnumerable<Variable> GetConstantArrays()
        {
            return ConstantArrays;
        }

        public IEnumerable<Variable> GetPrivateArrays()
        {
            return PrivateArrays;
        }

        public IEnumerable<Variable> GetGlobalAndGroupSharedArrays(bool includeDisabled)
        {
            List<Variable> all = new List<Variable>();
            all.AddRange(GlobalArrays);
            all.AddRange(GroupSharedArrays);
            return FilterDisabled(all, includeDisabled);
        }

        public IEnumerable<Variable> GetReadOnlyGlobalAndGroupSharedArrays(bool includeDisabled)
        {
            return FilterDisabled(ReadOnlyGlobalAndGroupSharedArrays, includeDisabled);
        }

        public IEnumerable<Variable> GetAtomicallyAccessedArrays(bool includeDisabled)
        {
          return FilterDisabled(AtomicallyAccessedArrays, includeDisabled);
        }

        public IEnumerable<Variable> GetAllArrays(bool includeDisabled)
        {
            List<Variable> all = new List<Variable>();
            all.AddRange(GetGlobalAndGroupSharedArrays(includeDisabled));
            all.AddRange(GetConstantArrays());
            all.AddRange(PrivateArrays);
            // Filtering here is not strictly necessary since it is done inside GetGlobalAndGroupSharedArrays
            return FilterDisabled(all, includeDisabled);
        }

        public bool ContainsGlobalOrGroupSharedArray(Variable v, bool includeDisabled)
        {
            return GetGlobalAndGroupSharedArrays(includeDisabled).Contains(v);
        }

        public bool ContainsConstantArray(Variable v)
        {
            return ConstantArrays.Contains(v);
        }

        public bool ContainsPrivateArray(Variable v)
        {
            return PrivateArrays.Contains(v);
        }

        public void AddGlobalArray(Variable v) {
          Debug.Assert(!GlobalArrays.Contains(v));
          GlobalArrays.Add(v);
        }

        public void AddGroupSharedArray(Variable v) {
          Debug.Assert(!GroupSharedArrays.Contains(v));
          GroupSharedArrays.Add(v);
        }

        public void AddPrivateArray(Variable v) {
          Debug.Assert(!PrivateArrays.Contains(v));
          PrivateArrays.Add(v);
        }

        public void AddConstantArray(Variable v) {
          Debug.Assert(!ConstantArrays.Contains(v));
          ConstantArrays.Add(v);
        }

        public void AddAtomicallyAccessedArray(Variable v) {
          Debug.Assert(!AtomicallyAccessedArrays.Contains(v));
          AtomicallyAccessedArrays.Add(v);
        }

        public void AddReadOnlyGlobalOrGroupSharedArray(Variable v) {
          Debug.Assert(GlobalArrays.Contains(v) || GroupSharedArrays.Contains(v));
          Debug.Assert(!ReadOnlyGlobalAndGroupSharedArrays.Contains(v));
          ReadOnlyGlobalAndGroupSharedArrays.Add(v);
        }

        public void RemovePrivateArray(Variable v) {
          Debug.Assert(PrivateArrays.Contains(v));
          PrivateArrays.Remove(v);
        }

        public void DisableGlobalOrGroupSharedArray(Variable v) {
          Debug.Assert(GetGlobalAndGroupSharedArrays(true).Contains(v));
          Debug.Assert(!DisabledArrays.Contains(v));
          DisabledArrays.Add(v);
        }

    }
}
