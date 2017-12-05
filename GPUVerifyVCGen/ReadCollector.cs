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
    using System.Diagnostics;
    using Microsoft.Boogie;

    public class ReadCollector : AccessCollector
    {
        public List<AccessRecord> NonPrivateAccesses { get; } = new List<AccessRecord>();

        public List<AccessRecord> PrivateAccesses { get; } = new List<AccessRecord>();

        public ReadCollector(IKernelArrayInfo state)
            : base(state)
        {
        }

        public override AssignLhs VisitSimpleAssignLhs(SimpleAssignLhs node)
        {
            return node;
        }

        public override Expr VisitNAryExpr(NAryExpr node)
        {
            if (node.Fun is MapSelect)
            {
                if ((node.Fun as MapSelect).Arity > 1)
                {
                    MultiDimensionalMapError();
                }

                if (!(node.Args[0] is IdentifierExpr))
                {
                  // This should only happen if the map is one of the special _USED maps for atomics
                  var nodeArgs0 = node.Args[0] as NAryExpr;
                  Debug.Assert(nodeArgs0 != null);
                  Debug.Assert(nodeArgs0.Fun is MapSelect);
                  Debug.Assert(nodeArgs0.Args[0] is IdentifierExpr);
                  Debug.Assert(((IdentifierExpr)nodeArgs0.Args[0]).Name.StartsWith("_USED"));
                  return base.VisitNAryExpr(node);
                }

                Debug.Assert(node.Args[0] is IdentifierExpr);
                var readVariable = (node.Args[0] as IdentifierExpr).Decl;
                var index = node.Args[1];
                this.VisitExpr(node.Args[1]);

                if (State.ContainsGlobalOrGroupSharedArray(readVariable, true))
                  NonPrivateAccesses.Add(new AccessRecord(readVariable, index));
                else if (State.ContainsPrivateArray(readVariable))
                  PrivateAccesses.Add(new AccessRecord(readVariable, index));

                return node;
            }
            else
            {
                return base.VisitNAryExpr(node);
            }
        }
    }
}
