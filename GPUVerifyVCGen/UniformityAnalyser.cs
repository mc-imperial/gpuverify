﻿namespace GPUVerify
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using Microsoft.Boogie;
    using Microsoft.Boogie.GraphUtil;

    public class UniformityAnalyser
    {
        private Program prog;

        private bool doAnalysis;

        private ISet<Implementation> entryPoints;

        private IEnumerable<Variable> nonUniformVars;

        private bool procedureChanged;

        private Dictionary<string, KeyValuePair<bool, Dictionary<string, bool>>> uniformityInfo;

        private Dictionary<string, HashSet<int>> nonUniformLoops;

        private Dictionary<string, HashSet<Block>> nonUniformBlocks;

        private Dictionary<string, HashSet<int>> loopsWithNonuniformReturn;

        private Dictionary<string, List<string>> inParameters;

        private Dictionary<string, List<string>> outParameters;

        /// <summary>
        /// Simplifies the CFG of the given implementation impl by merging each
        /// basic block with a single predecessor into that predecessor if the
        /// predecessor has a single successor.  If a uniformity analyser is
        /// being used then blocks will only be merged if they are both uniform
        /// or both non-uniform
        /// </summary>
        public static void MergeBlocksIntoPredecessors(Program prog, Implementation impl, UniformityAnalyser uni)
        {
            var blockGraph = prog.ProcessLoops(impl);
            var predMap = new Dictionary<Block, Block>();
            foreach (var block in blockGraph.Nodes)
            {
                try
                {
                    var pred = blockGraph.Predecessors(block).Single();
                    if (blockGraph.Successors(pred).Single() == block &&
                        (uni == null ||
                        (uni.IsUniform(impl.Name, pred) && uni.IsUniform(impl.Name, block)) ||
                        (!uni.IsUniform(impl.Name, pred) && !uni.IsUniform(impl.Name, block))))
                    {
                        Block predMapping;
                        while (predMap.TryGetValue(pred, out predMapping))
                            pred = predMapping;
                        pred.Cmds.AddRange(block.Cmds);
                        pred.TransferCmd = block.TransferCmd;
                        impl.Blocks.Remove(block);
                        predMap[block] = pred;
                    }

                    // If Single throws an exception above (i.e. not exactly one pred/succ), skip this block.
                }
                catch (InvalidOperationException)
                {
                }
            }
        }

        public UniformityAnalyser(Program prog, bool doAnalysis, ISet<Implementation> entryPoints, IEnumerable<Variable> nonUniformVars)
        {
            this.prog = prog;
            this.doAnalysis = doAnalysis;
            this.entryPoints = entryPoints;
            this.nonUniformVars = nonUniformVars;
            uniformityInfo = new Dictionary<string, KeyValuePair<bool, Dictionary<string, bool>>>();
            nonUniformLoops = new Dictionary<string, HashSet<int>>();
            nonUniformBlocks = new Dictionary<string, HashSet<Block>>();
            loopsWithNonuniformReturn = new Dictionary<string, HashSet<int>>();
            inParameters = new Dictionary<string, List<string>>();
            outParameters = new Dictionary<string, List<string>>();
        }

        public void Analyse()
        {
            var impls = prog.Implementations;

            foreach (var impl in impls)
            {
                bool uniformProcedure = doAnalysis || entryPoints.Contains(impl);

                uniformityInfo.Add(impl.Name, new KeyValuePair<bool, Dictionary<string, bool>>(uniformProcedure, new Dictionary<string, bool>()));

                nonUniformLoops.Add(impl.Name, new HashSet<int>());
                loopsWithNonuniformReturn.Add(impl.Name, new HashSet<int>());

                foreach (var v in nonUniformVars)
                    SetNonUniform(impl.Name, v.Name);

                foreach (Variable v in impl.LocVars)
                {
                    if (doAnalysis)
                    {
                        SetUniform(impl.Name, v.Name);
                    }
                    else
                    {
                        SetNonUniform(impl.Name, v.Name);
                    }
                }

                inParameters[impl.Name] = new List<string>();

                foreach (Variable v in impl.InParams)
                {
                    inParameters[impl.Name].Add(v.Name);
                    if (doAnalysis)
                    {
                        SetUniform(impl.Name, v.Name);
                    }
                    else
                    {
                        SetNonUniform(impl.Name, v.Name);
                    }
                }

                outParameters[impl.Name] = new List<string>();
                foreach (Variable v in impl.OutParams)
                {
                    outParameters[impl.Name].Add(v.Name);
                    if (doAnalysis)
                    {
                        SetUniform(impl.Name, v.Name);
                    }
                    else
                    {
                        SetNonUniform(impl.Name, v.Name);
                    }
                }

                procedureChanged = true;
            }

            var procs = prog.Procedures;

            foreach (var proc in procs)
            {
              if (uniformityInfo.ContainsKey(proc.Name))
              {
                continue;
              }

              bool uniformProcedure = doAnalysis;

              uniformityInfo.Add(proc.Name, new KeyValuePair<bool, Dictionary<string, bool>>(uniformProcedure, new Dictionary<string, bool>()));

              inParameters[proc.Name] = new List<string>();

              foreach (Variable v in proc.InParams)
              {
                inParameters[proc.Name].Add(v.Name);
                if (doAnalysis)
                {
                  SetUniform(proc.Name, v.Name);
                }
                else
                {
                  SetNonUniform(proc.Name, v.Name);
                }
              }

              outParameters[proc.Name] = new List<string>();
              foreach (Variable v in proc.OutParams)
              {
                outParameters[proc.Name].Add(v.Name);

                // We do not have a body for the procedure,
                // so we must assume it produces non-uniform
                // results
                SetNonUniform(proc.Name, v.Name);
              }

              procedureChanged = true;
            }

            if (doAnalysis)
            {
                while (procedureChanged)
                {
                    procedureChanged = false;

                    foreach (var impl in impls)
                    {
                        Analyse(impl, uniformityInfo[impl.Name].Key);
                    }
                }
            }

            foreach (var proc in procs)
            {
                if (!IsUniform(proc.Name))
                {
                    List<string> newIns = new List<string>();
                    newIns.Add("_P");
                    foreach (string s in inParameters[proc.Name])
                    {
                        newIns.Add(s);
                    }

                    foreach (string s in outParameters[proc.Name])
                    {
                        newIns.Add("_V" + s);
                    }

                    inParameters[proc.Name] = newIns;
                }
            }
        }

        private void Analyse(Implementation impl, bool controlFlowIsUniform)
        {
            if (!controlFlowIsUniform)
            {
                nonUniformBlocks[impl.Name] = new HashSet<Block>(impl.Blocks);

                foreach (Variable v in impl.LocVars)
                {
                    if (IsUniform(impl.Name, v.Name))
                    {
                        SetNonUniform(impl.Name, v.Name);
                    }
                }

                foreach (Variable v in impl.InParams)
                {
                    if (IsUniform(impl.Name, v.Name))
                    {
                        SetNonUniform(impl.Name, v.Name);
                    }
                }

                foreach (Variable v in impl.OutParams)
                {
                  if (IsUniform(impl.Name, v.Name))
                  {
                      SetNonUniform(impl.Name, v.Name);
                  }
                }

                foreach (Block b in impl.Blocks)
                {
                  Analyse(impl, b.Cmds, false);
                }

                return;
            }

            Graph<Block> blockGraph = prog.ProcessLoops(impl);
            var ctrlDep = blockGraph.ControlDependence();

            // Compute transitive closure of control dependence info.
            ctrlDep.TransitiveClosure();

            var nonUniformBlockSet = new HashSet<Block>();
            nonUniformBlocks[impl.Name] = nonUniformBlockSet;

            bool changed;
            do
            {
              changed = false;
              foreach (var block in impl.Blocks)
              {
                bool uniform = !nonUniformBlockSet.Contains(block);
                bool newUniform = Analyse(impl, block.Cmds, uniform);
                if (uniform && !newUniform)
                {
                  changed = true;
                  nonUniformBlockSet.Add(block);
                  Block pred = blockGraph.Predecessors(block).Single();
                  if (ctrlDep.ContainsKey(pred))
                    nonUniformBlockSet.UnionWith(ctrlDep[pred]);
                }
              }
            }
            while (changed);
        }

        private Procedure GetProcedure(string procedureName)
        {
            foreach (var p in prog.Procedures)
            {
                if (p.Name == procedureName)
                {
                    return p;
                }
            }

            Debug.Assert(false);
            return null;
        }

        private bool Analyse(Implementation impl, List<Cmd> cmdSeq, bool controlFlowIsUniform)
        {
            foreach (Cmd c in cmdSeq)
            {
                if (c is AssignCmd)
                {
                    AssignCmd assignCmd = c as AssignCmd;
                    foreach (var a in assignCmd.Lhss.Zip(assignCmd.Rhss))
                    {
                        if (a.Item1 is SimpleAssignLhs)
                        {
                            SimpleAssignLhs lhs = a.Item1 as SimpleAssignLhs;
                            Expr rhs = a.Item2;
                            if (IsUniform(impl.Name, lhs.AssignedVariable.Name) &&
                                (!controlFlowIsUniform || !IsUniform(impl.Name, rhs)))
                            {
                                SetNonUniform(impl.Name, lhs.AssignedVariable.Name);
                            }
                        }
                    }
                }
                else if (c is HavocCmd)
                {
                    HavocCmd havocCmd = c as HavocCmd;
                    foreach (IdentifierExpr ie in havocCmd.Vars)
                    {
                        if (IsUniform(impl.Name, ie.Decl.Name))
                        {
                            SetNonUniform(impl.Name, ie.Decl.Name);
                        }
                    }
                }
                else if (c is CallCmd)
                {
                    CallCmd callCmd = c as CallCmd;
                    DeclWithFormals callee = GetProcedure(callCmd.callee);
                    Debug.Assert(callee != null);

                    if (!controlFlowIsUniform)
                    {
                        if (IsUniform(callCmd.callee))
                        {
                            SetNonUniform(callCmd.callee);
                        }
                    }

                    for (int i = 0; i < callee.InParams.Count; i++)
                    {
                        if (IsUniform(callCmd.callee, callee.InParams[i].Name)
                            && !IsUniform(impl.Name, callCmd.Ins[i]))
                        {
                            SetNonUniform(callCmd.callee, callee.InParams[i].Name);
                        }
                    }

                    for (int i = 0; i < callee.OutParams.Count; i++)
                    {
                        if (IsUniform(impl.Name, callCmd.Outs[i].Name)
                        && !IsUniform(callCmd.callee, callee.OutParams[i].Name))
                        {
                            SetNonUniform(impl.Name, callCmd.Outs[i].Name);
                        }
                    }
                }
                else if (c is AssumeCmd)
                {
                    var ac = (AssumeCmd)c;
                    if (controlFlowIsUniform && QKeyValue.FindBoolAttribute(ac.Attributes, "partition") &&
                        !IsUniform(impl.Name, ac.Expr))
                    {
                      controlFlowIsUniform = false;
                    }
                }
            }

            return controlFlowIsUniform;
        }

        private int GetLoopId(WhileCmd wc)
        {
            AssertCmd inv = wc.Invariants[0] as AssertCmd;
            Debug.Assert(inv.Attributes.Key.Contains("loophead_"));
            return Convert.ToInt32(inv.Attributes.Key.Substring("loophead_".Length));
        }

        private void SetNonUniform(string procedureName)
        {
            uniformityInfo[procedureName] = new KeyValuePair<bool, Dictionary<string, bool>>(false, uniformityInfo[procedureName].Value);
            RecordProcedureChanged();
        }

        private void SetNonUniform(string procedureName, WhileCmd wc)
        {
            nonUniformLoops[procedureName].Add(GetLoopId(wc));
            RecordProcedureChanged();
        }

        public bool IsUniform(string procedureName)
        {
            if (!uniformityInfo.ContainsKey(procedureName))
            {
                return false;
            }

            return uniformityInfo[procedureName].Key;
        }

        public bool IsUniform(string procedureName, Block b)
        {
            if (!nonUniformBlocks.ContainsKey(procedureName))
            {
                return false;
            }

            return !nonUniformBlocks[procedureName].Contains(b);
        }

        private class UniformExpressionAnalysisVisitor : ReadOnlyVisitor
        {
          private bool isUniform = true;
          private Dictionary<string, bool> uniformityInfo;

          public UniformExpressionAnalysisVisitor(Dictionary<string, bool> uniformityInfo)
          {
            this.uniformityInfo = uniformityInfo;
          }

          public override Variable VisitVariable(Variable v)
          {
            if (!uniformityInfo.ContainsKey(v.Name))
            {
              isUniform = isUniform && (v is Constant);
            }
            else if (!uniformityInfo[v.Name])
            {
              isUniform = false;
            }

            return v;
          }

          internal bool IsUniform()
          {
            return isUniform;
          }
        }

        public bool IsUniform(string procedureName, Expr expr)
        {
            if (!uniformityInfo.ContainsKey(procedureName))
            {
                return false;
            }

            UniformExpressionAnalysisVisitor visitor = new UniformExpressionAnalysisVisitor(uniformityInfo[procedureName].Value);
            visitor.VisitExpr(expr);
            return visitor.IsUniform();
        }

        public bool IsUniform(string procedureName, string v)
        {
            if (!uniformityInfo.ContainsKey(procedureName))
            {
                return false;
            }

            if (!uniformityInfo[procedureName].Value.ContainsKey(v))
            {
                return false;
            }

            return uniformityInfo[procedureName].Value[v];
        }

        private void SetUniform(string procedureName, string v)
        {
            uniformityInfo[procedureName].Value[v] = true;
            RecordProcedureChanged();
        }

        private void RecordProcedureChanged()
        {
            procedureChanged = true;
        }

        private void SetNonUniform(string procedureName, string v)
        {
            uniformityInfo[procedureName].Value[v] = false;
            RecordProcedureChanged();
        }

        public void Dump()
        {
            foreach (string p in uniformityInfo.Keys)
            {
                Console.WriteLine("Procedure " + p + ": "
                    + (uniformityInfo[p].Key ? "uniform" : "nonuniform"));
                foreach (string v in uniformityInfo[p].Value.Keys)
                {
                    Console.WriteLine("  " + v + ": " +
                        (uniformityInfo[p].Value[v] ? "uniform" : "nonuniform"));
                }

                Console.Write("Ins [");
                for (int i = 0; i < inParameters[p].Count; i++)
                {
                    Console.Write((i == 0 ? string.Empty : ", ") + inParameters[p][i]);
                }

                Console.WriteLine("]");
                Console.Write("Outs [");
                for (int i = 0; i < outParameters[p].Count; i++)
                {
                    Console.Write((i == 0 ? string.Empty : ", ") + outParameters[p][i]);
                }

                Console.WriteLine("]");
                if (nonUniformLoops.ContainsKey(p))
                {
                  Console.Write("Non-uniform loops:");
                    foreach (int l in nonUniformLoops[p])
                    {
                      Console.Write(" " + l);
                    }

                  Console.WriteLine();
                }

                if (nonUniformBlocks.ContainsKey(p))
                {
                Console.Write("Non-uniform blocks:");
                  foreach (Block b in nonUniformBlocks[p])
                  {
                    Console.Write(" " + b.Label);
                }

                Console.WriteLine();
            }
        }
        }

        public string GetInParameter(string procName, int i)
        {
            return inParameters[procName][i];
        }

        public string GetOutParameter(string procName, int i)
        {
            return outParameters[procName][i];
        }

        public bool KnowsOf(string p)
        {
            return uniformityInfo.ContainsKey(p);
        }

        public void AddNonUniform(string proc, string v)
        {
            if (uniformityInfo.ContainsKey(proc))
            {
                Debug.Assert(!uniformityInfo[proc].Value.ContainsKey(v));
                uniformityInfo[proc].Value[v] = false;
            }
        }

        public void AddNonUniform(string proc, Block b)
        {
          if (nonUniformBlocks.ContainsKey(proc))
          {
            Debug.Assert(!nonUniformBlocks[proc].Contains(b));
            nonUniformBlocks[proc].Add(b);
          }
        }
    }
}
