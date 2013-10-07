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

namespace GPUVerify
{
    class ArrayControlFlowAnalyser
    {
        private GPUVerifier verifier;

        private bool ProcedureChanged;

        private Dictionary<string, Dictionary<string, HashSet<string>>> mayBeDerivedFrom;

        private HashSet<string> arraysWhichMayAffectControlFlow;

        public ArrayControlFlowAnalyser(GPUVerifier verifier)
        {
            this.verifier = verifier;
            mayBeDerivedFrom = new Dictionary<string, Dictionary<string, HashSet<string>>>();
            arraysWhichMayAffectControlFlow = new HashSet<string>();
        }

        internal void Analyse()
        {
            foreach (Declaration D in verifier.Program.TopLevelDeclarations)
            {
                if (D is Implementation)
                {
                    Implementation Impl = D as Implementation;

                    if(!mayBeDerivedFrom.ContainsKey(Impl.Name)) {
                      mayBeDerivedFrom.Add(Impl.Name, new Dictionary<string, HashSet<string>>());
                    }

                    SetNotDerivedFromSharedState(Impl.Name, GPUVerifier._X.Name);
                    SetNotDerivedFromSharedState(Impl.Name, GPUVerifier._Y.Name);
                    SetNotDerivedFromSharedState(Impl.Name, GPUVerifier._Z.Name);

                    foreach (Variable v in Impl.LocVars)
                    {
                        SetNotDerivedFromSharedState(Impl.Name, v.Name);
                    }

                    ProcedureChanged = true;
                }

                if (D is Procedure) {

                  Procedure Proc = D as Procedure;

                  if (!mayBeDerivedFrom.ContainsKey(Proc.Name)) {
                    mayBeDerivedFrom.Add(Proc.Name, new Dictionary<string, HashSet<string>>());
                  }

                  foreach (Variable v in verifier.KernelArrayInfo.getAllNonLocalArrays()) {
                    SetMayBeDerivedFrom(Proc.Name, v.Name, v.Name);
                  }

                  foreach (Variable v in Proc.InParams) {
                    SetNotDerivedFromSharedState(Proc.Name, v.Name);
                  }

                  foreach (Variable v in Proc.OutParams) {
                    SetNotDerivedFromSharedState(Proc.Name, v.Name);
                  }

                  foreach(Requires r in Proc.Requires) {
                    ExprMayAffectControlFlow(Proc.Name, r.Condition);
                  }

                  foreach (Ensures e in Proc.Ensures) {
                    ExprMayAffectControlFlow(Proc.Name, e.Condition);
                  }

                  foreach (Expr m in Proc.Modifies) {
                    ExprMayAffectControlFlow(Proc.Name, m);
                  }

                  ProcedureChanged = true;
                }

            }

            while (ProcedureChanged)
            {
                ProcedureChanged = false;

                foreach (Declaration D in verifier.Program.TopLevelDeclarations)
                {
                    if (D is Implementation)
                    {
                        Implementation Impl = D as Implementation;
                        Analyse(Impl);
                    }
                }
            }

            if (GPUVerifyVCGenCommandLineOptions.ShowArrayControlFlowAnalysis)
            {
                dump();
            }
        }

        private void SetNotDerivedFromSharedState(string p, string v)
        {
            mayBeDerivedFrom[p][v] = new HashSet<string>();
        }

        private void SetMayBeDerivedFrom(string p, string v, string w)
        {
            if (!mayBeDerivedFrom[p].ContainsKey(v))
            {
                mayBeDerivedFrom[p][v] = new HashSet<string>();
            }
            Debug.Assert(!mayBeDerivedFrom[p][v].Contains(w));
            mayBeDerivedFrom[p][v].Add(w);
            ProcedureChanged = true;
        }

        private void Analyse(Implementation Impl)
        {
            foreach (var b in Impl.Blocks)
                Analyse(Impl, b.Cmds);
        }

        private void Analyse(Implementation impl, StmtList stmtList)
        {
            foreach (BigBlock bb in stmtList.BigBlocks)
            {
                Analyse(impl, bb);
            }
        }

        private void ExprMayAffectControlFlow(string proc, Expr e)
        {
            var visitor = new VariablesOccurringInExpressionVisitor();
            visitor.VisitExpr(e);
            foreach (Variable v in visitor.GetVariables())
            {
                if (!mayBeDerivedFrom[proc].ContainsKey(v.Name))
                {
                    continue;
                }
                foreach (string s in mayBeDerivedFrom[proc][v.Name])
                {
                    if (!arraysWhichMayAffectControlFlow.Contains(s))
                    {
                        SetArrayMayAffectControlFlow(s);
                    }
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
                                foreach (String s in mayBeDerivedFrom[impl.Name][v.Name])
                                {
                                    if (!mayBeDerivedFrom[impl.Name][lhs.AssignedVariable.Name].Contains(s))
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
                        QKeyValue.FindBoolAttribute(callCmd.Proc.Attributes, "binary_barrier_invariant")) {
                        foreach (Expr param in callCmd.Ins) {
                            ExprMayAffectControlFlow(impl.Name, param);
                        }
                    } else if(callCmd.callee != verifier.BarrierProcedure.Name) {

                        Implementation CalleeImplementation = verifier.GetImplementation(callCmd.callee);
                        if (CalleeImplementation != null) {
                          for (int i = 0; i < CalleeImplementation.InParams.Count(); i++) {
                            VariablesOccurringInExpressionVisitor visitor = new VariablesOccurringInExpressionVisitor();
                            visitor.VisitExpr(callCmd.Ins[i]);

                            foreach (Variable v in visitor.GetVariables()) {
                              if (!mayBeDerivedFrom[impl.Name].ContainsKey(v.Name)) {
                                continue;
                              }


                              foreach (String s in mayBeDerivedFrom[impl.Name][v.Name]) {
                                if (!mayBeDerivedFrom[callCmd.callee][CalleeImplementation.InParams[i].Name].Contains(s)) {
                                  SetMayBeDerivedFrom(callCmd.callee, CalleeImplementation.InParams[i].Name, s);
                                }
                              }
                            }

                          }

                          for (int i = 0; i < CalleeImplementation.OutParams.Count(); i++) {
                            foreach (String s in mayBeDerivedFrom[callCmd.callee][CalleeImplementation.OutParams[i].Name]) {
                              if (!mayBeDerivedFrom[impl.Name][callCmd.Outs[i].Name].Contains(s)) {
                                SetMayBeDerivedFrom(impl.Name, callCmd.Outs[i].Name, s);
                              }
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
                else if (c is AssertCmd) {
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
                {
                    Analyse(impl, ifCmd.elseBlock);
                }
                Debug.Assert(ifCmd.elseIf == null);
            }

        }

        private void SetArrayMayAffectControlFlow(string s)
        {
            Debug.Assert(!arraysWhichMayAffectControlFlow.Contains(s));
            arraysWhichMayAffectControlFlow.Add(s);
            ProcedureChanged = true;
        }

        private void dump()
        {
            foreach (string s in arraysWhichMayAffectControlFlow)
            {
                Console.WriteLine("Array " + s + " may affect control flow");
            }

        }

        internal bool MayAffectControlFlow(string v)
        {
            return arraysWhichMayAffectControlFlow.Contains(v);
        }
    }
}
