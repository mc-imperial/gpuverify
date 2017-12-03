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
    using System.Linq;
    using Microsoft.Boogie;

    internal class NoAccessInstrumenter : INoAccessInstrumenter
    {
        private GPUVerifier verifier;
        private IKernelArrayInfo stateToCheck;

        public NoAccessInstrumenter(GPUVerifier verifier)
        {
            this.verifier = verifier;
            stateToCheck = verifier.KernelArrayInfo;
        }

        public void AddNoAccessInstrumentation()
        {
            foreach (var impl in verifier.Program.Implementations.ToList())
            {
                AddNoAccessAssumes(impl);
            }
        }

        private void AddNoAccessAssumes(Implementation impl)
        {
            impl.Blocks = impl.Blocks.Select(AddNoAccessAssumes).ToList();
        }

        private Block AddNoAccessAssumes(Block b)
        {
            b.Cmds = AddNoAccessAssumes(b.Cmds);
            return b;
        }

        private List<Cmd> AddNoAccessAssumes(List<Cmd> cs)
        {
            var result = new List<Cmd>();
            foreach (Cmd c in cs)
            {
                result.Add(c);
                if (c is AssignCmd)
                {
                    AssignCmd assign = c as AssignCmd;

                    ReadCollector rc = new ReadCollector(stateToCheck);
                    foreach (var rhs in assign.Rhss)
                        rc.Visit(rhs);
                    if (rc.nonPrivateAccesses.Count > 0)
                    {
                        foreach (AccessRecord ar in rc.nonPrivateAccesses)
                        {
                            AddNoAccessAssumes(result, ar);
                        }
                    }

                    foreach (var lhsRhs in assign.Lhss.Zip(assign.Rhss))
                    {
                        WriteCollector wc = new WriteCollector(stateToCheck);
                        wc.Visit(lhsRhs.Item1);
                        if (wc.FoundNonPrivateWrite())
                        {
                            AddNoAccessAssumes(result, wc.GetAccess());
                        }
                    }
                }
            }

            return result;
        }

        private void AddNoAccessAssumes(List<Cmd> result, AccessRecord ar)
        {
            result.Add(new AssumeCmd(Token.NoToken, Expr.Neq(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateNotAccessedVariable(ar.v.Name, ar.Index.Type)), ar.Index)));
        }
    }
}
