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

    public class UninterpretedFunctionRemover
    {
        public void Eliminate(Program program)
        {
            foreach (var impl in program.Implementations)
            {
                var cfg = Program.GraphFromImpl(impl);
                var visitor = new UFRemoverVisitor(program);
                foreach (var b in impl.Blocks)
                {
                    visitor.NewBlock();
                    b.Cmds = visitor.VisitCmdSeq(b.Cmds);
                    if (visitor.UFTemps.Last().Count() > 0)
                    {
                        foreach (var p in cfg.Predecessors(b))
                        {
                            p.Cmds.Add(new HavocCmd(
                                Token.NoToken,
                                visitor.UFTemps.Last().Select(item => new IdentifierExpr(Token.NoToken, item)).ToList()));
                        }
                    }
                }

                impl.LocVars.AddRange(visitor.UFTemps.SelectMany(item => item));
            }

            var newDecls = program.TopLevelDeclarations
                .Where(item => !(item is Function) || IsInterpreted(item as Function, program));
            program.ClearTopLevelDeclarations();
            program.AddTopLevelDeclarations(newDecls);
        }

        private static bool IsInterpreted(Function fun, Program prog)
        {
            if (fun.Body != null
                || QKeyValue.FindStringAttribute(fun.Attributes, "bvbuiltin") != null
                || QKeyValue.FindBoolAttribute(fun.Attributes, "constructor"))
            {
                return true;
            }

            if (fun.Name.Contains('#'))
            {
                return true;
            }

            return prog.TopLevelDeclarations.OfType<Axiom>().Any(item => UsesFun(item, fun));
        }

        private static bool UsesFun(Axiom axiom, Function fun)
        {
            var visitor = new FunctionIsReferencedVisitor(fun);
            visitor.VisitAxiom(axiom);
            return visitor.Found;
        }

        private class UFRemoverVisitor : Duplicator
        {
            private Program prog;
            private int counter = 0;

            public List<HashSet<LocalVariable>> UFTemps { get; }
                = new List<HashSet<LocalVariable>>();

            public UFRemoverVisitor(Program prog)
            {
                this.prog = prog;
            }

            public void NewBlock()
            {
                UFTemps.Add(new HashSet<LocalVariable>());
            }

            public override Expr VisitNAryExpr(NAryExpr node)
            {
                var funCall = node.Fun as FunctionCall;
                if (funCall == null || IsInterpreted(funCall.Func, prog))
                {
                    return base.VisitNAryExpr(node);
                }

                LocalVariable ufTemp = new LocalVariable(
                    Token.NoToken, new TypedIdent(Token.NoToken, "_UF_temp_" + counter, node.Type));
                counter++;
                UFTemps.Last().Add(ufTemp);
                return new IdentifierExpr(Token.NoToken, ufTemp);
            }
        }

        private class FunctionIsReferencedVisitor : StandardVisitor
        {
            private Function fun;

            public bool Found { get; private set; } = false;

            public FunctionIsReferencedVisitor(Function fun)
            {
                this.fun = fun;
            }

            public override Expr VisitNAryExpr(NAryExpr node)
            {
                if (node.Fun is FunctionCall && ((FunctionCall)node.Fun).Func.Name.Equals(fun.Name))
                    Found = true;

                return base.VisitNAryExpr(node);
            }
        }
    }
}
