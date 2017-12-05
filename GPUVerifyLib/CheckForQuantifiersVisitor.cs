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
    using Microsoft.Boogie;

    public class CheckForQuantifiersVisitor : StandardVisitor
    {
        private bool quantifiersExist = false;

        private CheckForQuantifiersVisitor()
        {
        }

        public static bool Find(Program node)
        {
            var cfq = new CheckForQuantifiersVisitor();
            cfq.VisitProgram(node);
            return cfq.quantifiersExist;
        }

        public override QuantifierExpr VisitQuantifierExpr(QuantifierExpr node)
        {
            node = base.VisitQuantifierExpr(node);
            quantifiersExist = true;
            return node;
        }
    }
}
