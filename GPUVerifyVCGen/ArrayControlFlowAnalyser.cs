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
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Microsoft.Boogie;

    internal class ArrayControlFlowAnalyser
    {
        private GPUVerifier verifier;

        private bool procedureChanged;

        private Dictionary<string, Dictionary<string, HashSet<string>>> mayBeDerivedFrom;

        private HashSet<string> arraysWhichMayAffectControlFlow;

        public ArrayControlFlowAnalyser(GPUVerifier verifier)
        {
            this.verifier = verifier;
            mayBeDerivedFrom = new Dictionary<string, Dictionary<string, HashSet<string>>>();
            arraysWhichMayAffectControlFlow = new HashSet<string>();
        }

        public void Analyse()
        {
            foreach (Declaration decl in verifier.Program.TopLevelDeclarations)
            {
                if (decl is Implementation)
                {
                    Implementation impl = decl as Implementation;

                    if (!mayBeDerivedFrom.ContainsKey(impl.Name))
                        mayBeDerivedFrom.Add(impl.Name, new Dictionary<string, HashSet<string>>());

                    SetNotDerivedFromSharedState(impl.Name, GPUVerifier._X.Name);
                    SetNotDerivedFromSharedState(impl.Name, GPUVerifier._Y.Name);
                    SetNotDerivedFromSharedState(impl.Name, GPUVerifier._Z.Name);

                    foreach (Variable v in impl.LocVars)
                        SetNotDerivedFromSharedState(impl.Name, v.Name);

                    procedureChanged = true;
                }

                if (decl is Procedure)
                {
                    Procedure proc = decl as Procedure;

                    if (!mayBeDerivedFrom.ContainsKey(proc.Name))
                        mayBeDerivedFrom.Add(proc.Name, new Dictionary<string, HashSet<string>>());

                    foreach (Variable v in verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(true))
                        SetMayBeDerivedFrom(proc.Name, v.Name, v.Name);

                    foreach (Variable v in proc.InParams)
                        SetNotDerivedFromSharedState(proc.Name, v.Name);

                    foreach (Variable v in proc.OutParams)
                        SetNotDerivedFromSharedState(proc.Name, v.Name);

                    foreach (Requires r in proc.Requires)
                        ExprMayAffectControlFlow(proc.Name, r.Condition);

                    foreach (Ensures e in proc.Ensures)
                        ExprMayAffectControlFlow(proc.Name, e.Condition);

                    foreach (Expr m in proc.Modifies)
                        ExprMayAffectControlFlow(proc.Name, m);

                    procedureChanged = true;
                }
            }

            while (procedureChanged)
            {
                procedureChanged = false;

                foreach (Declaration decl in verifier.Program.TopLevelDeclarations)
                {
                    if (decl is Implementation)
                    {
                        Implementation impl = decl as Implementation;
                        Analyse(impl);
                    }
                }
            }

            if (GPUVerifyVCGenCommandLineOptions.ShowArrayControlFlowAnalysis)
                Dump();
        }

        public bool MayAffectControlFlow(string v)
        {
            return arraysWhichMayAffectControlFlow.Contains(v);
        }

        private void SetNotDerivedFromSharedState(string p, string v)
        {
            mayBeDerivedFrom[p][v] = new HashSet<string>();
        }

        private void SetMayBeDerivedFrom(string p, string v, string w)
        {
            if (!mayBeDerivedFrom[p].ContainsKey(v))
                mayBeDerivedFrom[p][v] = new HashSet<string>();

            Debug.Assert(!mayBeDerivedFrom[p][v].Contains(w));
            mayBeDerivedFrom[p][v].Add(w);
            procedureChanged = true;
        }

        private void Analyse(Implementation impl)
        {
            foreach (var b in impl.Blocks)
                Analyse(impl, b.Cmds);
        }

        private void Analyse(Implementation impl, StmtList stmtList)
        {
            foreach (BigBlock bb in stmtList.BigBlocks)
                Analyse(impl, bb);
        }

        private void ExprMayAffectControlFlow(string proc, Expr e)
        {
            var visitor = new VariablesOccurringInExpressionVisitor();
            visitor.VisitExpr(e);
            foreach (Variable v in visitor.GetVariables())
            {
                if (!mayBeDerivedFrom[proc].ContainsKey(v.Name))
                    continue;

                foreach (string s in mayBeDerivedFrom[proc][v.Name])
                {
                    if (!arraysWhichMayAffectControlFlow.Contains(s))
                        SetArrayMayAffectControlFlow(s);
                }
            }
        }

        private void Analyse(Implementation impl, List<Cmd> cs)
        {
            foreach (var c in cs)
            {
                if (c is AssignCmd)
                {
                    AssignCmd assignCmd = c as AssignCmd;
                    for (int i = 0; i != assignCmd.Lhss.Count; i++)
                    {
                        if (assignCmd.Lhss[i] is SimpleAssignLhs)
                        {
                            SimpleAssignLhs lhs = assignCmd.Lhss[i] as SimpleAssignLhs;
                            Expr rhs = assignCmd.Rhss[i];

                            VariablesOccurringInExpressionVisitor visitor = new VariablesOccurringInExpressionVisitor();
                            visitor.VisitExpr(rhs);

                            foreach (Variable v in visitor.GetVariables())
                            {
                                if (!mayBeDerivedFrom[impl.Name].ContainsKey(v.Name))
                                {
                                    continue;
                                }

                                foreach (string s in mayBeDerivedFrom[impl.Name][v.Name])
                                {
                                    if (mayBeDerivedFrom[impl.Name].ContainsKey(lhs.AssignedVariable.Name) && !mayBeDerivedFrom[impl.Name][lhs.AssignedVariable.Name].Contains(s))
                                    {
                                        SetMayBeDerivedFrom(impl.Name, lhs.AssignedVariable.Name, s);
                                    }
                                }
                            }
                        }
                    }
                }
                else if (c is CallCmd)
                {
                    CallCmd callCmd = c as CallCmd;

                    if (QKeyValue.FindBoolAttribute(callCmd.Proc.Attributes, "barrier_invariant") ||
                        QKeyValue.FindBoolAttribute(callCmd.Proc.Attributes, "binary_barrier_invariant"))
                    {
                        foreach (Expr param in callCmd.Ins)
                            ExprMayAffectControlFlow(impl.Name, param);
                    }
                    else if (!GPUVerifier.IsBarrier(callCmd.Proc))
                    {
                        Implementation calleeImplementation = verifier.GetImplementation(callCmd.callee);
                        if (calleeImplementation != null)
                        {
                            for (int i = 0; i < calleeImplementation.InParams.Count(); i++)
                            {
                                VariablesOccurringInExpressionVisitor visitor = new VariablesOccurringInExpressionVisitor();
                                visitor.VisitExpr(callCmd.Ins[i]);

                                foreach (Variable v in visitor.GetVariables())
                                {
                                    if (!mayBeDerivedFrom[impl.Name].ContainsKey(v.Name))
                                        continue;

                                    foreach (string s in mayBeDerivedFrom[impl.Name][v.Name])
                                    {
                                        if (!mayBeDerivedFrom[callCmd.callee][calleeImplementation.InParams[i].Name].Contains(s))
                                            SetMayBeDerivedFrom(callCmd.callee, calleeImplementation.InParams[i].Name, s);
                                    }
                                }
                            }

                            for (int i = 0; i < calleeImplementation.OutParams.Count(); i++)
                            {
                                foreach (string s in mayBeDerivedFrom[callCmd.callee][calleeImplementation.OutParams[i].Name])
                                {
                                    if (!mayBeDerivedFrom[impl.Name][callCmd.Outs[i].Name].Contains(s))
                                        SetMayBeDerivedFrom(impl.Name, callCmd.Outs[i].Name, s);
                                }
                            }
                        }
                    }
                }
                else if (c is AssumeCmd)
                {
                    var assumeCmd = c as AssumeCmd;
                    ExprMayAffectControlFlow(impl.Name, assumeCmd.Expr);
                }
                else if (c is AssertCmd)
                {
                    var assertCmd = c as AssertCmd;
                    ExprMayAffectControlFlow(impl.Name, assertCmd.Expr);
                }
            }
        }

        private void Analyse(Implementation impl, BigBlock bb)
        {
            Analyse(impl, bb.simpleCmds);

            if (bb.ec is WhileCmd)
            {
                WhileCmd wc = bb.ec as WhileCmd;

                ExprMayAffectControlFlow(impl.Name, wc.Guard);

                Analyse(impl, wc.Body);
            }
            else if (bb.ec is IfCmd)
            {
                IfCmd ifCmd = bb.ec as IfCmd;

                ExprMayAffectControlFlow(impl.Name, ifCmd.Guard);

                Analyse(impl, ifCmd.thn);
                if (ifCmd.elseBlock != null)
                    Analyse(impl, ifCmd.elseBlock);

                Debug.Assert(ifCmd.elseIf == null);
            }
        }

        private void SetArrayMayAffectControlFlow(string s)
        {
            Debug.Assert(!arraysWhichMayAffectControlFlow.Contains(s));
            arraysWhichMayAffectControlFlow.Add(s);
            procedureChanged = true;
        }

        private void Dump()
        {
            foreach (string s in arraysWhichMayAffectControlFlow)
            {
                Console.WriteLine("Array " + s + " may affect control flow");
            }
        }
    }
}
