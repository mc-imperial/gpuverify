//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


using System;
using Microsoft.Boogie;

namespace GPUVerify
{
  public class CheckForQuantifiers : StandardVisitor
  {
    bool quantifiersExist = false;

    private CheckForQuantifiers() { }

    public override QuantifierExpr VisitQuantifierExpr(QuantifierExpr node)
    {
      node = base.VisitQuantifierExpr(node);
      quantifiersExist = true;
      return node;
    }

    public static bool Found(Program node)
    {
      var cfq = new CheckForQuantifiers();
      cfq.VisitProgram(node);
      return cfq.quantifiersExist;
    }
  }

  internal class VariableFinderVisitor : StandardVisitor
  {
    private string VarName;
    private Variable Variable = null;

    internal VariableFinderVisitor(string VarName) {
      this.VarName = VarName;
    }

    public override Variable VisitVariable(Variable node) {
      if (node.Name.Equals(VarName)) {
        Variable = node;
      }
      return base.VisitVariable(node);
    }

    internal Variable GetVariable() {
      return Variable;
    }
  }
}

