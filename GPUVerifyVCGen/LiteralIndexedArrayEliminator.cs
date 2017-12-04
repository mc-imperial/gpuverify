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
    using Microsoft.Boogie;

    internal class LiteralIndexedArrayEliminator
    {
        private GPUVerifier verifier;
        private Dictionary<string, GlobalVariable> arrayCache = new Dictionary<string, GlobalVariable>();

        public LiteralIndexedArrayEliminator(GPUVerifier verifier)
        {
            this.verifier = verifier;
        }

        public void Eliminate(Program program)
        {
            var arrays = CollectRelevantArrays(program);
            RemoveArraysFromProgram(program, arrays);
            ReplaceArraysUsesWithVariables(program, arrays);
        }

        private void ReplaceArraysUsesWithVariables(Program program, Dictionary<string, HashSet<string>> arrays)
        {
            foreach (var b in program.Blocks())
            {
                b.Cmds = new EliminatorVisitor(arrays, this).VisitCmdSeq(b.Cmds);
            }

            foreach (var p in program.TopLevelDeclarations.OfType<Procedure>())
            {
                p.Requires = new EliminatorVisitor(arrays, this).VisitRequiresSeq(p.Requires);
                p.Ensures = new EliminatorVisitor(arrays, this).VisitEnsuresSeq(p.Ensures);
            }
        }

        private void RemoveArraysFromProgram(Program program, Dictionary<string, HashSet<string>> arrays)
        {
            foreach (var a in verifier.KernelArrayInfo.GetPrivateArrays().ToList())
            {
                if (arrays.ContainsKey(a.Name))
                {
                    verifier.KernelArrayInfo.RemovePrivateArray(a);
                    program.RemoveTopLevelDeclarations(x => x == a);

                    foreach (var l in arrays[a.Name])
                    {
                        program.AddTopLevelDeclaration(MakeVariableForArrayIndex(a, l));
                    }
                }
            }
        }

        internal GlobalVariable MakeVariableForArrayIndex(Variable array, string literal)
        {
            var arrayName = array.Name + "$" + literal;
            if (!arrayCache.ContainsKey(arrayName))
            {
                arrayCache[arrayName] = new GlobalVariable(
                    array.tok, new TypedIdent(array.tok, arrayName, (array.TypedIdent.Type as MapType).Result));
            }

            return arrayCache[arrayName];
        }

        private Dictionary<string, HashSet<string>> CollectRelevantArrays(Program program)
        {
            var collector = new LiteralIndexVisitor(verifier);
            collector.VisitProgram(program);
            return collector.LiteralIndexedArrays;
        }
    }

    internal class LiteralIndexVisitor : StandardVisitor
    {
        // Maps an array to a set of strings, each of which denotes a literal with which the array can be indexed.
        internal readonly Dictionary<string, HashSet<string>> LiteralIndexedArrays;

        internal LiteralIndexVisitor(GPUVerifier verifier)
        {
            this.LiteralIndexedArrays = new Dictionary<string, HashSet<string>>();
            foreach (var v in verifier.KernelArrayInfo.GetPrivateArrays())
            {
                this.LiteralIndexedArrays[v.Name] = new HashSet<string>();
            }
        }

        public override Expr VisitNAryExpr(NAryExpr node)
        {
            if (node.Fun is MapSelect && node.Args.Count() == 2)
            {
                var map = node.Args[0] as IdentifierExpr;
                if (map != null)
                {
                    if (LiteralIndexedArrays.ContainsKey(map.Name))
                    {
                        UpdateIndexingInfo(node.Args[1], map.Name);
                    }
                }
            }

            return base.VisitNAryExpr(node);
        }

        public override Cmd VisitAssignCmd(AssignCmd node)
        {
            foreach (var lhs in node.Lhss.OfType<MapAssignLhs>())
            {
                if (!(lhs.Map is SimpleAssignLhs))
                {
                    continue;
                }

                if (lhs.Indexes.Count() != 1)
                {
                    continue;
                }

                var map = (lhs.Map as SimpleAssignLhs).AssignedVariable;
                if (LiteralIndexedArrays.ContainsKey(map.Name))
                {
                    UpdateIndexingInfo(lhs.Indexes[0], map.Name);
                }
            }

            return base.VisitAssignCmd(node);
        }

        private void UpdateIndexingInfo(Expr maybeLiteral, string mapName)
        {
            if (maybeLiteral is LiteralExpr)
            {
                LiteralIndexedArrays[mapName].Add(maybeLiteral.ToString());
            }
            else
            {
                // The array is not always indexed by a literal
                LiteralIndexedArrays.Remove(mapName);
            }
        }
    }

    internal class EliminatorVisitor : Duplicator
    {
        private Dictionary<string, HashSet<string>> arrays;
        private LiteralIndexedArrayEliminator arrayEliminator;

        public EliminatorVisitor(Dictionary<string, HashSet<string>> arrays, LiteralIndexedArrayEliminator arrayEliminator)
        {
            this.arrays = arrays;
            this.arrayEliminator = arrayEliminator;
        }

        public override Expr VisitNAryExpr(NAryExpr node)
        {
            if (node.Fun is MapSelect && node.Args.Count() == 2)
            {
                var map = node.Args[0] as IdentifierExpr;
                if (map != null)
                {
                    if (arrays.ContainsKey(map.Name))
                    {
                        Debug.Assert(node.Args[1] is LiteralExpr);
                        return new IdentifierExpr(
                            Token.NoToken,
                            arrayEliminator.MakeVariableForArrayIndex(map.Decl, node.Args[1].ToString()));
                    }
                }
            }

            return base.VisitNAryExpr(node);
        }

        private AssignLhs TransformLhs(AssignLhs lhs)
        {
            var mapLhs = lhs as MapAssignLhs;
            if (mapLhs == null
              || !(mapLhs.Map is SimpleAssignLhs)
              || mapLhs.Indexes.Count() != 1)
            {
                return (AssignLhs)Visit(lhs);
            }

            var map = (mapLhs.Map as SimpleAssignLhs).AssignedVariable;

            if (!arrays.ContainsKey(map.Name))
            {
                return (AssignLhs)Visit(lhs);
            }

            Debug.Assert(mapLhs.Indexes[0] is LiteralExpr);

            var lhsId = new IdentifierExpr(
                Token.NoToken,
                arrayEliminator.MakeVariableForArrayIndex(map.Decl, mapLhs.Indexes[0].ToString()));
            return new SimpleAssignLhs(lhs.tok, lhsId);
        }

        public override Cmd VisitAssignCmd(AssignCmd node)
        {
            return new AssignCmd(
                node.tok,
                node.Lhss.Select(item => TransformLhs(item)).ToList(),
                node.Rhss.Select(item => VisitExpr(item)).ToList());
        }
    }
}
