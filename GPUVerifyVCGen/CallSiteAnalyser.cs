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

    public class CallSiteAnalyser
    {
        private GPUVerifier verifier;
        private Dictionary<Procedure, List<CallCmd>> callSites;

        public CallSiteAnalyser(GPUVerifier verifier)
        {
            this.verifier = verifier;
            callSites = new Dictionary<Procedure, List<CallCmd>>();
        }

        public void Analyse()
        {
            FindAllCallSites();
            LiteralArgumentAnalyser();
        }

        private void FindAllCallSites()
        {
            foreach (Declaration decl in verifier.Program.TopLevelDeclarations)
            {
                if (decl is Implementation)
                    FindCallSites(decl as Implementation);
            }
        }

        private void FindCallSites(Implementation impl)
        {
            FindCallSites(impl.Blocks);
        }

        private void FindCallSites(List<Block> blocks)
        {
            foreach (Block b in blocks)
                FindCallSites(b);
        }

        private void FindCallSites(Block b)
        {
            FindCallSites(b.Cmds);
        }

        private void FindCallSites(List<Cmd> cs)
        {
            foreach (Cmd c in cs)
            {
                if (c is CallCmd)
                {
                    CallCmd callCmd = c as CallCmd;

                    if (!callSites.ContainsKey(callCmd.Proc))
                    {
                        callSites[callCmd.Proc] = new List<CallCmd>();
                    }

                    callSites[callCmd.Proc].Add(callCmd);
                }
            }
        }

        private void LiteralArgumentAnalyser()
        {
            foreach (Procedure p in callSites.Keys)
            {
                for (int i = 0; i < p.InParams.Count(); i++)
                    LiteralArgumentAnalyser(p, i);
            }
        }

        private void LiteralArgumentAnalyser(Procedure p, int arg)
        {
            LiteralExpr literal = null;

            foreach (CallCmd callCmd in callSites[p])
            {
                if (callCmd.Ins[arg] == null || !(callCmd.Ins[arg] is LiteralExpr))
                    return;

                LiteralExpr l = callCmd.Ins[arg] as LiteralExpr;

                if (literal == null)
                    literal = l;
                else if (!literal.Equals(l))
                    return;
            }

            Expr e;
            e = new IdentifierExpr(Token.NoToken, p.InParams[arg]);
            e = Expr.Eq(e, literal);
            p.Requires.Add(new Requires(false, e));
        }
    }
}
