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

    class WriteCollector : AccessCollector
    {
        private AccessRecord access = null;
        private bool isPrivate;

        public WriteCollector(IKernelArrayInfo state)
            : base(state)
        {
        }

        private bool NoWrittenVariable()
        {
            return access == null;
        }

        public override AssignLhs VisitMapAssignLhs(MapAssignLhs node)
        {
            Debug.Assert(NoWrittenVariable());

            if (!State.ContainsGlobalOrGroupSharedArray(node.DeepAssignedVariable, true)
                  && !State.ContainsPrivateArray(node.DeepAssignedVariable))
            {
                return node;
            }

            Variable writtenVariable = node.DeepAssignedVariable;

            CheckMapIndex(node);
            Debug.Assert(!(node.Map is MapAssignLhs));

            access = new AccessRecord(writtenVariable, node.Indexes[0]);

            isPrivate = State.ContainsPrivateArray(writtenVariable);

            return node;
        }

        private void CheckMapIndex(MapAssignLhs node)
        {
            if (node.Indexes.Count > 1)
            {
                MultiDimensionalMapError();
            }
        }

        internal AccessRecord GetAccess()
        {
            return access;
        }

        internal bool FoundPrivateWrite()
        {
            return access != null && isPrivate;
        }

        internal bool FoundNonPrivateWrite()
        {
            return access != null && !isPrivate;
        }
    }
}
