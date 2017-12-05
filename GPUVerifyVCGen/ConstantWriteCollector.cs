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
    using System.Diagnostics;
    using Microsoft.Boogie;

    public class ConstantWriteCollector : AccessCollector
    {
        private AccessRecord access = null;

        public ConstantWriteCollector(IKernelArrayInfo state)
            : base(state)
        {
        }

        public override AssignLhs VisitMapAssignLhs(MapAssignLhs node)
        {
            Debug.Assert(NoWrittenVariable());

            if (!State.ContainsConstantArray(node.DeepAssignedVariable))
                return node;

            Variable writtenVariable = node.DeepAssignedVariable;

            CheckMapIndex(node);
            Debug.Assert(!(node.Map is MapAssignLhs));

            access = new AccessRecord(writtenVariable, node.Indexes[0]);

            return node;
        }

        public bool FoundWrite()
        {
            return access != null;
        }

        public AccessRecord GetAccess()
        {
            return access;
        }

        private bool NoWrittenVariable()
        {
            return access == null;
        }

        private void CheckMapIndex(MapAssignLhs node)
        {
            if (node.Indexes.Count > 1)
                MultiDimensionalMapError();
        }
    }
}
