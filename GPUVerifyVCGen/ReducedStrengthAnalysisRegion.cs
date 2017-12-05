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
    using System.Linq;
    using Microsoft.Boogie;
    using Microsoft.Boogie.GraphUtil;

    using VarDefMap =
        System.Collections.Generic.Dictionary<
            Microsoft.Boogie.Variable,
            System.Collections.Generic.List<System.Tuple<IRegion, Microsoft.Boogie.Expr>>>;

    public class ReducedStrengthAnalysisRegion
    {
        private Implementation impl;
        private GPUVerifier verifier;
        private Dictionary<object, Dictionary<string, ModStrideConstraint>> strideConstraintMap
            = new Dictionary<object, Dictionary<string, ModStrideConstraint>>();

        private ReducedStrengthAnalysisRegion(Implementation i, GPUVerifier v)
        {
            impl = i;
            verifier = v;
        }

        private void AddAssignment(IRegion region, VarDefMap multiDefMap, SimpleAssignLhs lhs, Expr rhs)
        {
            if (lhs == null)
                return;

            var variable = lhs.DeepAssignedVariable;
            if (!multiDefMap.ContainsKey(variable))
                multiDefMap[variable] = new List<Tuple<IRegion, Expr>>();
            multiDefMap[variable].Add(new Tuple<IRegion, Expr>(region, rhs));
        }

        private void FindAssignments(IRegion region, VarDefMap multiDefMap)
        {
            foreach (var c in region.CmdsChildRegions())
            {
                var aCmd = c as AssignCmd;
                if (aCmd != null)
                {
                    foreach (var a in aCmd.Lhss.Zip(aCmd.Rhss))
                        AddAssignment(region, multiDefMap, a.Item1 as SimpleAssignLhs, a.Item2);
                }

                var child = c as IRegion;
                if (child != null)
                    FindAssignments(child, multiDefMap);
            }
        }

        private bool OrderDefs(List<Tuple<IRegion, Expr>> defs, Graph<Block> cfg)
        {
            if (defs[1].Item1.Header() == null)
                defs.Reverse();

            if (defs[0].Item1.Header() == null)
                return defs[1].Item1.Header() != null;

            if (cfg.DominatorMap.DominatedBy(defs[0].Item1.Header(), defs[1].Item1.Header()))
                defs.Reverse();

            return defs[0].Item1.Header() != defs[1].Item1.Header();
        }

        private class StrideForm
        {
            public StrideFormKind Kind { get; private set; }

            public Expr Op { get; private set; }

            public StrideForm(StrideFormKind kind)
            {
                this.Kind = kind;
                this.Op = null;
            }

            public StrideForm(StrideFormKind kind, Expr op)
            {
                this.Kind = kind;
                this.Op = op;
            }

            public enum StrideFormKind
            {
                Bottom,
                Identity,
                Constant,
                Product
            }

            public static StrideForm ComputeStrideForm(Variable v, Expr e, GPUVerifier verifier, HashSet<Variable> modSet)
            {
                Expr lhs, rhs;

                if (e is LiteralExpr)
                    return new StrideForm(StrideFormKind.Constant, e);

                var ie = e as IdentifierExpr;
                if (ie != null)
                {
                    if (ie.Decl is Constant)
                        return new StrideForm(StrideFormKind.Constant, e);
                    else if (ie.Decl == v)
                        return new StrideForm(StrideFormKind.Identity, e);
                    else if (!modSet.Contains(ie.Decl))
                        return new StrideForm(StrideFormKind.Constant, e);
                }

                if (verifier.IntRep.IsAdd(e, out lhs, out rhs))
                {
                    var lhssf = ComputeStrideForm(v, lhs, verifier, modSet);
                    var rhssf = ComputeStrideForm(v, rhs, verifier, modSet);
                    if (lhssf.Kind == StrideFormKind.Constant && rhssf.Kind == StrideFormKind.Constant)
                        return new StrideForm(StrideFormKind.Constant, e);
                    else if (lhssf.Kind == StrideFormKind.Constant && rhssf.Kind == StrideFormKind.Identity)
                        return new StrideForm(StrideFormKind.Product, lhs);
                    else if (lhssf.Kind == StrideFormKind.Identity && rhssf.Kind == StrideFormKind.Constant)
                        return new StrideForm(StrideFormKind.Product, rhs);
                    else if (lhssf.Kind == StrideFormKind.Constant && rhssf.Kind == StrideFormKind.Product)
                        return new StrideForm(StrideFormKind.Product, verifier.IntRep.MakeAdd(lhs, rhssf.Op));
                    else if (lhssf.Kind == StrideFormKind.Product && rhssf.Kind == StrideFormKind.Constant)
                        return new StrideForm(StrideFormKind.Product, verifier.IntRep.MakeAdd(lhssf.Op, rhs));
                    else
                        return new StrideForm(StrideFormKind.Bottom);
                }

                var ne = e as NAryExpr;
                if (ne != null)
                {
                    foreach (Expr op in ne.Args)
                    {
                        if (ComputeStrideForm(v, op, verifier, modSet).Kind != StrideFormKind.Constant)
                            return new StrideForm(StrideFormKind.Bottom);
                    }

                    return new StrideForm(StrideFormKind.Constant, e);
                }

                return new StrideForm(StrideFormKind.Bottom);
            }
        }

        private void AddDefinitionPair(Variable v, Expr defInd, Expr defInit, object regionId, HashSet<Variable> modSet)
        {
            var sf = StrideForm.ComputeStrideForm(v, defInd, verifier, modSet);
            if (sf.Kind != StrideForm.StrideFormKind.Product)
                return;

            var sc = new ModStrideConstraint(sf.Op, defInit);
            if (sc.IsBottom())
                return;

            if (!strideConstraintMap.ContainsKey(regionId))
                strideConstraintMap[regionId] = new Dictionary<string, ModStrideConstraint>();

            strideConstraintMap[regionId][v.Name] = sc;
        }

        private void AnalyseAssignment(Variable variable, List<Tuple<IRegion, Expr>> defs, Graph<Block> cfg, HashSet<Variable> modSet)
        {
            // Ensure defs[0] refers to the definition from the outermost region
            // This also checks that the definitions are in different regions
            if (!OrderDefs(defs, cfg))
                return;

            var regionId = defs[1].Item1.Identifier();
            var varDefAnalysis = verifier.VarDefAnalysesRegion[impl];
            var varDef = varDefAnalysis.GetPossibleInductionVariableDefintion(variable.Name, regionId);
            if (varDef == null)
                return;

            HashSet<string> loopFreeVars;
            var defInd = varDefAnalysis.SubstDefinitions(varDef, impl.Name, out loopFreeVars);
            if (loopFreeVars.Any(i => i != variable.Name))
                return;

            var modSetLoop = defs[1].Item1.GetModifiedVariables();
            var v = new VariablesOccurringInExpressionVisitor();
            v.Visit(defs[0].Item2);
            if (v.GetVariables().Intersect(modSetLoop).Any())
                return;

            AddDefinitionPair(variable, defInd, defs[0].Item2, regionId, modSet);
        }

        private void Analyse()
        {
            var cfg = verifier.Program.ProcessLoops(impl);
            var modSet = verifier.RootRegion(impl).GetModifiedVariables();
            var multiDefMap = new VarDefMap();
            FindAssignments(verifier.RootRegion(impl), multiDefMap);
            foreach (var e in multiDefMap.Where(i => i.Value.Count == 2))
                AnalyseAssignment(e.Key, e.Value, cfg, modSet);
        }

        public IEnumerable<string> StridedInductionVariables(object regionId)
        {
            if (!strideConstraintMap.ContainsKey(regionId))
                return Enumerable.Empty<string>();

            return strideConstraintMap[regionId].Keys;
        }

        public ModStrideConstraint GetStrideConstraint(string variable, object regionId)
        {
            if (!strideConstraintMap.ContainsKey(regionId))
            {
                return null;
            }
            else
            {
                int id;
                var strippedVariable = Utilities.StripThreadIdentifier(variable, out id);
                ModStrideConstraint msc;
                if (strideConstraintMap[regionId].TryGetValue(strippedVariable, out msc))
                {
                    return new ModStrideConstraint(
                        verifier.MaybeDualise(msc.Mod, id, impl.Name),
                        verifier.MaybeDualise(msc.ModEq, id, impl.Name));
                }
                else
                {
                    return null;
                }
            }
        }

        public ModStrideConstraint GetStrideConstraint(string variable)
        {
            foreach (var r in strideConstraintMap.Keys)
            {
                var msc = GetStrideConstraint(variable, r);
                if (msc != null)
                    return msc;
            }

            return null;
        }

        public static ReducedStrengthAnalysisRegion Analyse(Implementation impl, GPUVerifier verifier)
        {
            var a = new ReducedStrengthAnalysisRegion(impl, verifier);
            a.Analyse();
            return a;
        }
    }
}
