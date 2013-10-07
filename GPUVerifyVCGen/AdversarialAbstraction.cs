//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;
using System.Diagnostics.Contracts;

namespace GPUVerify {

  class AdversarialAbstraction {

    private GPUVerifier verifier;

    private List<Variable> NewLocalVars = null;
    private int AbstractedCallArgCounter = 0;

    internal AdversarialAbstraction(GPUVerifier verifier) {
      this.verifier = verifier;
    }

    internal void Abstract() {
      List<Declaration> NewTopLevelDeclarations = new List<Declaration>();
      foreach (Declaration d in verifier.Program.TopLevelDeclarations) {
        if (d is Variable &&
          verifier.KernelArrayInfo.ContainsNonLocalArray(d as Variable) &&
          verifier.ArrayModelledAdversarially(d as Variable)) {
          continue;
        }

        if (d is Implementation) {
          Abstract(d as Implementation);
        }

        if (d is Procedure) {
          Abstract(d as Procedure);
        }

        NewTopLevelDeclarations.Add(d);

      }

      verifier.Program.TopLevelDeclarations = NewTopLevelDeclarations;

    }

    private void AbstractRequiresClauses(Procedure proc) {
      List<Requires> newRequires = new List<Requires>();
      foreach (Requires r in proc.Requires) {
        var visitor = new AccessesAdversarialArrayVisitor(verifier);
        visitor.VisitRequires(r);
        if (!visitor.found) {
          newRequires.Add(r);
        }
      }
      proc.Requires = newRequires;
    }

    private void Abstract(Procedure proc) {
      AbstractModifiesSet(proc);
      AbstractRequiresClauses(proc);
    }

    private void AbstractModifiesSet(Procedure proc) {
      List<IdentifierExpr> NewModifies = new List<IdentifierExpr>();
      foreach (IdentifierExpr e in proc.Modifies) {
        var visitor = new AccessesAdversarialArrayVisitor(verifier);
        visitor.VisitIdentifierExpr(e);
        if(!visitor.found) {
          NewModifies.Add(e);
        }
      }
      proc.Modifies = NewModifies;
    }

    private void Abstract(Implementation impl) {
      NewLocalVars = new List<Variable>();
      AbstractedCallArgCounter = 0;
      foreach (Variable v in impl.LocVars) {
        Debug.Assert(!verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v));
        NewLocalVars.Add(v);
      }
      impl.LocVars = NewLocalVars;
      impl.Blocks = impl.Blocks.Select(Abstract).ToList();
      NewLocalVars = null;

    }

    private Block Abstract(Block b) {

      var NewCmds = new List<Cmd>();

      foreach (Cmd c in b.Cmds) {

        if (c is CallCmd) {
          var call = c as CallCmd;
          for(int i = 0; i < call.Ins.Count; i++) {
            ReadCollector rc = new ReadCollector(verifier.KernelArrayInfo);
            rc.Visit(call.Ins[i]);
            bool foundAdversarial = false;
            foreach (AccessRecord ar in rc.accesses) {
              if (verifier.ArrayModelledAdversarially(ar.v)) {
                foundAdversarial = true;
                break;
              }
            }

            if (foundAdversarial) {
              LocalVariable lv = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken,
                "_abstracted_call_arg_" + AbstractedCallArgCounter, call.Ins[i].Type));
              AbstractedCallArgCounter++;
              NewLocalVars.Add(lv);
              NewCmds.Add(new HavocCmd(Token.NoToken,
                new List<IdentifierExpr>(new IdentifierExpr[] { new IdentifierExpr(Token.NoToken, lv) })));
              call.Ins[i] = new IdentifierExpr(Token.NoToken, lv);
            }
          }
        }

        if (c is AssignCmd) {
          AssignCmd assign = c as AssignCmd;

          var lhss = new List<AssignLhs>();
          var rhss = new List<Expr>();

          foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss)) {
            AssignLhs lhs = LhsRhs.Item1;
            Expr rhs = LhsRhs.Item2;
            ReadCollector rc = new ReadCollector(verifier.KernelArrayInfo);
            rc.Visit(rhs);

            bool foundAdversarial = false;
            foreach (AccessRecord ar in rc.accesses) {
              if (verifier.ArrayModelledAdversarially(ar.v)) {
                foundAdversarial = true;
                break;
              }
            }

            if (foundAdversarial) {
              Debug.Assert(lhs is SimpleAssignLhs);
              NewCmds.Add(new HavocCmd(c.tok, new List<IdentifierExpr>(new IdentifierExpr[] { (lhs as SimpleAssignLhs).AssignedVariable })));
              continue;
            }

            WriteCollector wc = new WriteCollector(verifier.KernelArrayInfo);
            wc.Visit(lhs);
            if (wc.GetAccess() != null && verifier.ArrayModelledAdversarially(wc.GetAccess().v)) {
              continue; // Just remove the write
            }

            lhss.Add(lhs);
            rhss.Add(rhs);
          }

          if (lhss.Count != 0) {
            NewCmds.Add(new AssignCmd(assign.tok, lhss, rhss));
          }
          continue;
        }
        NewCmds.Add(c);
      }

      b.Cmds = NewCmds;
      return b;
    }

    class AccessesAdversarialArrayVisitor : StandardVisitor {
      internal bool found;
      private GPUVerifier verifier;

      internal AccessesAdversarialArrayVisitor(GPUVerifier verifier) {
        this.found = false;
        this.verifier = verifier;
      }

      public override Variable VisitVariable(Variable v) {
        if (verifier.KernelArrayInfo.ContainsNonLocalArray(v)) {
          if (verifier.ArrayModelledAdversarially(v)) {
            found = true;
          }
        }
        return base.VisitVariable(v);
      }

    }

  }


}
