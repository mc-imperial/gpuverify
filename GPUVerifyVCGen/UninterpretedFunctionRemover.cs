using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class UninterpretedFunctionRemover
  {
    private GPUVerifier verifier;

    public UninterpretedFunctionRemover(GPUVerifier verifier)
    {
      this.verifier = verifier;
    }

    internal void Eliminate(Program Program)
    {
      foreach(var impl in Program.Implementations()) {
        var CFG = Program.GraphFromImpl(impl);
        var visitor = new UFRemoverVisitor(Program);
        foreach(var b in impl.Blocks) {
          visitor.NewBlock();
          b.Cmds = visitor.VisitCmdSeq(b.Cmds);
          if(visitor.UFTemps.Last().Count() > 0) {
            foreach(var p in CFG.Predecessors(b)) {
              p.Cmds.Add(new HavocCmd(Token.NoToken, visitor.UFTemps.Last().Select(Item => new IdentifierExpr(Token.NoToken, Item)).ToList()));
            }
          }
        }
        impl.LocVars.AddRange(visitor.UFTemps.SelectMany(Item => Item));
      }

      Program.TopLevelDeclarations = Program.TopLevelDeclarations.Where(
        item => !(item is Function) || IsInterpreted(item as Function, Program)).ToList();
    }

    internal static bool IsInterpreted(Function fun, Program prog)
    {
      if(fun.Body != null ||
         QKeyValue.FindStringAttribute(fun.Attributes, "bvbuiltin") != null ||
         QKeyValue.FindBoolAttribute(fun.Attributes, "constructor")) {
        return true;
      }
      if(fun.Name.Contains('#')) {
        return true;
      }
      return prog.TopLevelDeclarations.OfType<Axiom>().Any(Item => UsesFun(Item, fun));
    }

    private static bool UsesFun(Axiom axiom, Function fun)
    {
      var visitor = new FunctionIsReferencedVisitor(fun);
      visitor.VisitAxiom(axiom);
      return visitor.Found();
    }

  }

  class UFRemoverVisitor : Duplicator {

    private Program prog;

    public UFRemoverVisitor(Program prog) {
      this.prog = prog;
    }

    public readonly 
      List<HashSet<LocalVariable>> UFTemps = new List<HashSet<LocalVariable>>();

    private int counter = 0;

    internal void NewBlock() {
      UFTemps.Add(new HashSet<LocalVariable>());
    }

    public override Expr VisitNAryExpr(NAryExpr node)
    {
      var FunCall = node.Fun as FunctionCall;
      if(FunCall == null || UninterpretedFunctionRemover.IsInterpreted(FunCall.Func, prog)) {
        return base.VisitNAryExpr(node);
      }
      LocalVariable UFTemp = new LocalVariable(Token.NoToken,
        new TypedIdent(Token.NoToken, "_UF_temp_" + counter, node.Type));
      counter++;
      UFTemps.Last().Add(UFTemp);
      return new IdentifierExpr(Token.NoToken, UFTemp);
    }

  }

  class FunctionIsReferencedVisitor : StandardVisitor {

    private Function fun;
    private bool found;

    public FunctionIsReferencedVisitor(Function fun)
    {
      this.fun = fun;
      this.found = false;
    }

    internal bool Found()
    {
      return found;
    }

    public override Expr VisitNAryExpr(NAryExpr node)
    {
      if(node.Fun is FunctionCall) {
        if(((FunctionCall)node.Fun).Func.Name.Equals(fun.Name)) {
          found = true;
        }
      }
      return base.VisitNAryExpr(node);
    }

  }


}
