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
    using System.Diagnostics;
    using System.Linq;
    using System.Text.RegularExpressions;
    using Microsoft.Boogie;
    using Microsoft.Boogie.GraphUtil;

    public class AssignmentExpressionExpander
    {
        private static readonly Regex TempVariable = new Regex("^v[0-9]+$");
        private static readonly Regex GpuVariable = new Regex("(local|global)_id_(x|y|z)$");
        private Graph<Block> cfg;
        private Variable initialVariable;
        private Expr unexpandedExpr = null;
        private HashSet<Variable> gpuVariables = new HashSet<Variable>();

        public AssignmentExpressionExpander(Graph<Block> cfg, Variable variable)
        {
            this.cfg = cfg;
            this.initialVariable = variable;
            DiscoverVariableAssignments(this.initialVariable);
        }

        public HashSet<Variable> GetGPUVariables()
        {
            return gpuVariables;
        }

        public Expr GetUnexpandedExpr()
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
                        foreach (var lhsRhs in lhss.Zip(assignCmd.Rhss))
                        {
                            if (lhsRhs.Item1.DeepAssignedVariable.Name == variable.Name)
                            {
                                assignment = lhsRhs.Item2;
                                if (variable.Name == this.initialVariable.Name)
                                    unexpandedExpr = lhsRhs.Item2;
                                goto AnalyseAssignment;
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
                                goto AnalyseAssignment;
                            }
                        }
                    }
                }
            }

        // Label used to allow exit from multiple levels in the previous loop nest
        AnalyseAssignment:
            if (assignment != null)
            {
                var visitor = new VariablesOccurringInExpressionVisitor();
                visitor.Visit(assignment);
                foreach (Variable discovered in visitor.GetVariables())
                {
                    if (TempVariable.IsMatch(discovered.Name) && discovered.Name != variable.Name)
                        DiscoverVariableAssignments(discovered);
                    else if (GpuVariable.IsMatch(discovered.Name))
                        gpuVariables.Add(discovered);
                }
            }
        }
    }
}
