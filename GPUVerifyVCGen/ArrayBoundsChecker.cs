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

    internal class ArrayBoundsChecker
    {
        private static int arraySourceID = 0;

        private GPUVerifier verifier;
        private Dictionary<string, Tuple<string, int>> arraySizes;

        private int currSourceLocNum = 0;

        private enum BOUND_TYPE
        {
            LOWER,
            UPPER
        }

        public ArrayBoundsChecker(GPUVerifier verifier, Program program)
        {
            this.verifier = verifier;
            this.arraySizes = GetBounds(program);
        }

        public Dictionary<string, Tuple<string, int>> GetBounds(Program program)
        {
            Dictionary<string, Tuple<string, int>> arraySizes = new Dictionary<string, Tuple<string, int>>();

            // Get Axioms with array_info attribute, save dictionary of array_info value to (source_name, source_dimensions, ?elem_width, source_elem_width?)
            foreach (Axiom a in program.Axioms.Where(item => QKeyValue.FindStringAttribute((item as Axiom).Attributes, "array_info") != null))
            {
                string arrayName = QKeyValue.FindStringAttribute(a.Attributes, "array_info");

                string arraySize = QKeyValue.FindStringAttribute(a.Attributes, "source_dimensions");
                if (arraySize.Equals("*"))
                    continue;

                string sourceArrayName = QKeyValue.FindStringAttribute(a.Attributes, "source_name");
                int elemWidth = QKeyValue.FindIntAttribute(a.Attributes, "elem_width", -1);
                int sourceElemWidth = QKeyValue.FindIntAttribute(a.Attributes, "source_elem_width", -1);
                int totalArraySize = ParseArraySize(arraySize) * sourceElemWidth / elemWidth;
                Tuple<string, int> arrayInfo = Tuple.Create<string, int>(sourceArrayName, totalArraySize);
                arraySizes.Add(arrayName, arrayInfo);
            }

            return arraySizes;
        }

        public void CheckBounds(Program program)
        {
            foreach (Implementation i in program.TopLevelDeclarations.ToList().Where(item => item.GetType() == typeof(Implementation)))
            {
                foreach (Block b in i.Blocks)
                {
                    var newCmds = new List<Cmd>();
                    foreach (Cmd c in b.Cmds)
                    {
                        if (c is AssertCmd && QKeyValue.FindBoolAttribute((c as AssertCmd).Attributes, "sourceloc"))
                        {
                            currSourceLocNum = QKeyValue.FindIntAttribute((c as AssertCmd).Attributes, "sourceloc_num", -1);
                        }
                        else if (c is AssignCmd)
                        {
                            List<AccessRecord> accesses = GetArrayAccesses(c as AssignCmd);
                            foreach (AccessRecord ar in accesses)
                            {
                                bool recordedArray = !arraySizes.ContainsKey(ar.v.Name); /* only gen lower bound check */
                                int arrSize = recordedArray ? 0 : arraySizes[ar.v.Name].Item2;
                                newCmds.AddRange(GenArrayBoundChecks(recordedArray, ar, arrSize));
                            }
                        }

                        newCmds.Add(c);
                    }

                    b.Cmds = newCmds;
                }
            }
        }

        private int ParseArraySize(string arraySize)
        {
            List<string> arrayDimensionSizes = new List<string>(arraySize.Split(','));
            int totalArraySize = 1;
            foreach (string size in arrayDimensionSizes)
                totalArraySize *= Convert.ToInt32(size);
            return totalArraySize;
        }

        private List<AccessRecord> GetArrayAccesses(AssignCmd assign)
        {
            List<AccessRecord> accesses = new List<AccessRecord>();
            ReadCollector rc = new ReadCollector(this.verifier.KernelArrayInfo);
            WriteCollector wc = new WriteCollector(this.verifier.KernelArrayInfo);

            foreach (var rhs in assign.Rhss)
            {
                rc.Visit(rhs);
            }

            foreach (AccessRecord ar in rc.nonPrivateAccesses.Union(rc.privateAccesses))
            {
                accesses.Add(ar);
            }

            foreach (var lhs in assign.Lhss)
            {
                wc.Visit(lhs);
            }

            if (wc.FoundNonPrivateWrite() || wc.FoundPrivateWrite())
                accesses.Add(wc.GetAccess());

            return accesses;
        }

        private List<Cmd> GenArrayBoundChecks(bool onlyLower, AccessRecord ar, int arrDim)
        {
            List<Cmd> boundChecks = new List<Cmd>();

            var arrayOffset = verifier.FindOrCreateArrayOffsetVariable(ar.v.Name);

            boundChecks.Add(new AssignCmd(
                Token.NoToken,
                new List<AssignLhs> { new SimpleAssignLhs(Token.NoToken, Expr.Ident(arrayOffset)) },
                new List<Expr> { ar.Index }));

            var key = new QKeyValue(Token.NoToken, "captureState", new List<object> { "bounds_check_state_" + arraySourceID }, null);
            key = new QKeyValue(Token.NoToken, "check_id", new List<object> { "bounds_check_state_" + arraySourceID }, key);
            key = new QKeyValue(Token.NoToken, "do_not_predicate", new List<object> { }, key);
            boundChecks.Add(new AssumeCmd(Token.NoToken, Expr.True, key));
            boundChecks.Add(GenBoundCheck(BOUND_TYPE.LOWER, ar, arrDim, arrayOffset));

            if (!onlyLower)
                boundChecks.Add(GenBoundCheck(BOUND_TYPE.UPPER, ar, arrDim, arrayOffset));

            arraySourceID++;
            return boundChecks;
        }

        private AssertCmd GenBoundCheck(BOUND_TYPE btype, AccessRecord ar, int arrDim, Variable offsetVar)
        {
            Expr boundExpr = null;
            IdentifierExpr offsetVarExpr = new IdentifierExpr(Token.NoToken, offsetVar);
            switch (btype)
            {
                case BOUND_TYPE.LOWER:
                    boundExpr = verifier.IntRep.MakeSge(offsetVarExpr, verifier.IntRep.GetLiteral(0, verifier.size_t_bits)); break;
                case BOUND_TYPE.UPPER:
                    boundExpr = verifier.IntRep.MakeSlt(offsetVarExpr, verifier.IntRep.GetLiteral(arrDim, verifier.size_t_bits)); break;
            }

            var key = new QKeyValue(Token.NoToken, "array_name", new List<object> { ar.v.Name }, null);
            key = new QKeyValue(Token.NoToken, "check_id", new List<object> { "bounds_check_state_" + arraySourceID }, key);
            key = new QKeyValue(Token.NoToken, "sourceloc_num", new List<object> { new LiteralExpr(Token.NoToken, Microsoft.Basetypes.BigNum.FromInt(currSourceLocNum)) }, key);
            key = new QKeyValue(Token.NoToken, "array_bounds", new List<object> { }, key);
            return new AssertCmd(Token.NoToken, boundExpr, key);
        }
    }
}
