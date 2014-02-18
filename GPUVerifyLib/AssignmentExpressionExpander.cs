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
  private static Regex TEMP_VARIABLE = new Regex("^v[0-9]+$");
  private static Regex GPU_VARIABLE = new Regex("(local|global)_id_(x|y|z)$");
  
  private Graph<Block> cfg;
  private Variable initialVariable;
  private Expr unexpandedExpr = null;
  private HashSet<Variable> GPUVariables = new HashSet<Variable>();
  
  public AssignmentExpressionExpander(Implementation impl, Variable variable)
  {
   this.cfg = Program.GraphFromImpl(impl); 
   this.initialVariable = variable;
   DiscoverVariableAssignments(this.initialVariable);
  }

  public AssignmentExpressionExpander(Graph<Block> cfg, Variable variable)
  {
   this.cfg = cfg;
   this.initialVariable = variable;
   DiscoverVariableAssignments(this.initialVariable);
  }
  
  public HashSet<Variable> GetGPUVariables ()
  {
   return GPUVariables;
  }
  
  public Expr GetUnexpandedExpr ()
  {
   Debug.Assert(unexpandedExpr != null);
   return unexpandedExpr;
  }

  private void DiscoverVariableAssignments(Variable variable)
  {
   Absy assignment = null;
   foreach (Block block in cfg.Nodes)
   {
    foreach (Cmd cmd in block.Cmds)
    {
     if (cmd is AssignCmd)
     {
      AssignCmd assignCmd = cmd as AssignCmd;
      var lhss = assignCmd.Lhss.OfType<SimpleAssignLhs>();
      foreach (var LhsRhs in lhss.Zip(assignCmd.Rhss))
      {
       if (LhsRhs.Item1.DeepAssignedVariable.Name == variable.Name)
       {
        assignment = LhsRhs.Item2;
        if (variable.Name == this.initialVariable.Name)
         unexpandedExpr = LhsRhs.Item2;
       }
      }
     }
     else if (cmd is CallCmd)
     {
      CallCmd call = cmd as CallCmd;
      foreach (IdentifierExpr outParam in call.Outs)
      {
       if (outParam.Name == variable.Name)
       {
        assignment = cmd;
       }
      }
     }
    }
   }
   
   Debug.Assert(assignment != null);
   var visitor = new VariablesOccurringInExpressionVisitor();
   visitor.Visit(assignment);
   foreach (Variable discovered in visitor.GetVariables())
   {
    if (TEMP_VARIABLE.IsMatch(discovered.Name) && discovered.Name != variable.Name)
     DiscoverVariableAssignments(discovered);
    else if (GPU_VARIABLE.IsMatch(discovered.Name))
     GPUVariables.Add(discovered); 
   }
  }  
 }
 
}

