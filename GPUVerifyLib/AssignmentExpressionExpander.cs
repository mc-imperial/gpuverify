//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Boogie;
using Microsoft.Boogie.GraphUtil;

namespace GPUVerify
{
 public class AssignmentExpressionExpander
 {
  private static Regex TEMP_VARIABLE = new Regex("v[0-9]+$");
  private Graph<Block> cfg;
  private Dictionary<Variable, Expr> assignments = new Dictionary<Variable, Expr>();
  private Variable initialVariable;
  private Expr expanded;

  public AssignmentExpressionExpander(Implementation impl, Variable variable)
  {
   this.cfg = Program.GraphFromImpl(impl); 
   this.initialVariable = variable;
   Compute();
  }

  public AssignmentExpressionExpander(Graph<Block> cfg, Variable variable)
  {
   this.cfg = cfg;
   this.initialVariable = variable;
   Compute();
  }

  public Expr GetUnexpandedExpr()
  {
   return assignments[this.initialVariable];
  }
  
  public Expr GetExpandedExpr ()
  {
   return expanded;
  }

  private void Compute()
  {
   DiscoverVariableAssignments(this.initialVariable);
   ExprDuplicator duplicator = new ExprDuplicator(assignments);
   expanded = duplicator.Visit(assignments[this.initialVariable]) as Expr;
  }

  private void DiscoverVariableAssignments(Variable variable)
  {
   foreach (Block block in cfg.Nodes)
   {
    foreach (AssignCmd assignment in block.Cmds.OfType<AssignCmd>())
    {
     var lhss = assignment.Lhss.OfType<SimpleAssignLhs>();
     foreach (var LhsRhs in lhss.Zip(assignment.Rhss))
     {
      if (LhsRhs.Item1.DeepAssignedVariable.Name == variable.Name)
      {
       Debug.Assert(!assignments.ContainsKey(variable));
       assignments[variable] = LhsRhs.Item2;
      }
     }
    }
   }
   var visitor = new VariablesOccurringInExpressionVisitor();
   visitor.Visit(assignments[variable]);
   foreach (Variable discovered in visitor.GetVariables())
   {
    if (TEMP_VARIABLE.IsMatch(discovered.Name))
     DiscoverVariableAssignments(discovered);
   }
  }

  class ExprDuplicator : Duplicator
  {
   private Dictionary<Variable, Expr> assignments;

   public ExprDuplicator(Dictionary<Variable, Expr> assignments)
   {
    this.assignments = assignments;
   }

   public override Absy Visit(Absy node)
   {
    if (node is NAryExpr)
    {
     NAryExpr _node = node as NAryExpr;
     if (_node.Fun is BinaryOperator)
     {
      Expr one = (Expr)Visit(_node.Args[0]);
      Expr two = (Expr)Visit(_node.Args[1]);
      Expr[] arguments = new Expr[] { one, two };
      return new NAryExpr(Token.NoToken, _node.Fun, new List<Expr>(arguments));
     }
     else if (_node.Fun is FunctionCall)
     {
      List<Expr> arguments = new List<Expr>();
      foreach (var arg in _node.Args)
      {
       arguments.Add((Expr)Visit(arg));
      }
      return new NAryExpr(Token.NoToken, _node.Fun, arguments);
     }
    }
    else if (node is IdentifierExpr)
    {
     IdentifierExpr _node = node as IdentifierExpr;
     if (assignments.ContainsKey(_node.Decl))
      return Visit(assignments[_node.Decl]);
     return new IdentifierExpr(Token.NoToken, _node.Decl);
    }
    return node;
   }
  }
  
 }
}

