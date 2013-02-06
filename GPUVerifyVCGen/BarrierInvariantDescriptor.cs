using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify {
  abstract class BarrierInvariantDescriptor {

    protected Expr Predicate;
    protected Expr BarrierInvariant;
    protected QKeyValue SourceLocationInfo;
    protected KernelDualiser Dualiser;
    protected string ProcName;
    protected List<Expr> AccessExprs;

    public BarrierInvariantDescriptor(Expr Predicate, Expr BarrierInvariant,
          QKeyValue SourceLocationInfo,
          KernelDualiser Dualiser, string ProcName) {
      this.Predicate = Predicate;
      this.BarrierInvariant = BarrierInvariant;
      this.SourceLocationInfo = SourceLocationInfo;
      this.Dualiser = Dualiser;
      this.ProcName = ProcName;
      this.AccessExprs = new List<Expr>();

      if (CommandLineOptions.BarrierAccessChecks) {
        var visitor = new SubExprVisitor();
        visitor.VisitExpr(this.BarrierInvariant);
        foreach (NAryExpr e in visitor.SubExprs) {
          var v = (e.Args[0] as IdentifierExpr);
          var index = e.Args[1];
          this.AccessExprs.Add(Expr.Imp(Predicate, BuildAccessedExpr(v.Name, index)));
        }
      }
    }

    internal abstract AssertCmd GetAssertCmd();

    internal abstract List<AssumeCmd> GetInstantiationCmds();

    protected Expr NonNegative(Expr e) {
      return Dualiser.verifier.MakeBVSge(
        e, GPUVerifier.ZeroBV());
    }

    protected Expr NotTooLarge(Expr e) {
      return Dualiser.verifier.MakeBVSlt(e,
        new IdentifierExpr(Token.NoToken, 
          Dualiser.verifier.GetGroupSize("X")));
    }

    private Expr BuildAccessedExpr(string name, Expr e) {
      return Expr.Neq(new IdentifierExpr(Token.NoToken, Dualiser.verifier.FindOrCreateNotAccessedVariable(name, e.Type)), e);
    }

    public QKeyValue GetSourceLocationInfo() {
      return SourceLocationInfo;
    }

    public List<Expr> GetAccessedExprs() {
      return AccessExprs;
    }

    class SubExprVisitor : StandardVisitor {
      internal HashSet<Expr> SubExprs;

      internal SubExprVisitor() {
        this.SubExprs = new HashSet<Expr>();
      }

      public override Expr VisitNAryExpr(NAryExpr node) {
        if (node.Fun is MapSelect) {
          Debug.Assert((node.Fun as MapSelect).Arity == 1);
          Debug.Assert(node.Args[0] is IdentifierExpr);
          var v = (node.Args[0] as IdentifierExpr).Decl;
          if (QKeyValue.FindBoolAttribute(v.Attributes, "group_shared") ||
              QKeyValue.FindBoolAttribute(v.Attributes, "global")) {
            SubExprs.Add(node);
          }
        }
        return base.VisitNAryExpr(node);
      }

    }

  }
}
