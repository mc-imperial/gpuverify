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
    using System.Text;
    using Microsoft.Basetypes;
    using Microsoft.Boogie;

    public class ExprTree : System.Collections.IEnumerable
    {
        private Dictionary<int, HashSet<Node>> levels = new Dictionary<int, HashSet<Node>>();
        private List<Node> nodes = new List<Node>();
        private Node root = null;
        private int height = 0;
        private Expr expr;

        public BitVector Evaluation { get; set; }

        public bool Initialised { get; set; } = true;

        public HashSet<string> OffsetVariables { get; private set; } = new HashSet<string>();

        public ExprTree(Expr expr)
        {
            this.expr = expr;
            root = Node.CreateFromExpr(expr);
            levels[0] = new HashSet<Node>();
            levels[0].Add(root);
            SetLevels(root, 0);

            // Set node IDs and see if any scalar node is a race offset variable
            int nodeID = 0;
            foreach (Node node in nodes)
            {
                node.ID = nodeID++;
                if (node is ScalarSymbolNode)
                {
                    ScalarSymbolNode scalarNode = node as ScalarSymbolNode;
                    if (BoogieInterpreter.RegularExpressions.OffsetVariable.IsMatch(scalarNode.Symbol))
                        OffsetVariables.Add(scalarNode.Symbol);

                    if (BoogieInterpreter.RegularExpressions.WatchdoVariable.IsMatch(scalarNode.Symbol))
                    {
                        var visitor = new VariablesOccurringInExpressionVisitor();
                        visitor.Visit(this.expr);
                        string offsetVariable = string.Empty;
                        foreach (Variable variable in visitor.GetVariables())
                        {
                            if (BoogieInterpreter.RegularExpressions.TrackingVariable.IsMatch(variable.Name))
                            {
                                int index = variable.Name.IndexOf('$');
                                string arrayName = variable.Name.Substring(index);
                                string accessType = variable.Name.Substring(0, index);
                                if (accessType == "_WRITE_HAS_OCCURRED_")
                                    offsetVariable = "_WRITE_OFFSET_" + arrayName;
                                else if (accessType == "_READ_HAS_OCCURRED_")
                                    offsetVariable = "_READ_OFFSET_" + arrayName;
                                else
                                    offsetVariable = "_ATOMIC_OFFSET_" + arrayName;
                                break;
                            }
                        }

                        OffsetVariables.Add(offsetVariable);
                    }
                }
            }
        }

        private void SetLevels(Node parent, int level)
        {
            nodes.Add(parent);
            int newLevel = level + 1;
            height = Math.Max(height, newLevel);
            if (!levels.ContainsKey(newLevel))
                levels[newLevel] = new HashSet<Node>();
            foreach (Node child in parent.GetChildren())
            {
                levels[newLevel].Add(child);
                SetLevels(child, newLevel);
            }
        }

        public void ClearState()
        {
            foreach (Node node in nodes)
            {
                if (!(node is LiteralNode))
                    node.ClearState();
            }

            Initialised = true;
        }

        public Node Root()
        {
            return root;
        }

        public System.Collections.IEnumerator GetEnumerator()
        {
            for (int i = height - 1; i >= 0; --i)
            {
                yield return levels[i];
            }
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < height; ++i)
            {
                builder.Append(string.Format("Level {0}", i)).Append(Environment.NewLine);

                foreach (Node node in levels[i])
                {
                    builder.Append(node.ID);
                    if (i > 0)
                        builder.Append(string.Format(" (parent = {0})  ", node.GetParent().ID));
                }

                builder.Append(Environment.NewLine);
            }

            for (int i = 0; i < height; ++i)
            {
                foreach (Node node in levels[i])
                {
                    builder.Append(node.ID + " " + node.ToString());
                    builder.Append(Environment.NewLine);
                }
            }

            return builder.ToString();
        }
    }

    public abstract class Node
    {
        public int ID { get; set; } = -1;

        public bool Initialised { get; set; } = true;

        private Node parent = null;

        protected List<Node> Children { get; private set; } = new List<Node>();

        public Node()
        {
        }

        public bool IsLeaf()
        {
            return Children.Count == 0;
        }

        public List<Node> GetChildren()
        {
            return Children;
        }

        public Node GetParent()
        {
            return parent;
        }

        public abstract void ClearState();

        public static Node CreateFromExpr(Expr expr)
        {
            if (expr is NAryExpr)
            {
                NAryExpr nary = expr as NAryExpr;
                if (nary.Fun is IfThenElse)
                {
                    Node one = CreateFromExpr(nary.Args[0]);
                    Node two = CreateFromExpr(nary.Args[1]);
                    Node three = CreateFromExpr(nary.Args[2]);
                    Node parent = new TernaryNode(nary.Fun.FunctionName, one, two, three);
                    one.parent = parent;
                    two.parent = parent;
                    three.parent = parent;
                    return parent;
                }
                else if (nary.Fun is BinaryOperator)
                {
                    Node one = CreateFromExpr(nary.Args[0]);
                    Node two = CreateFromExpr(nary.Args[1]);
                    Node parent = new BinaryNode(nary.Fun.FunctionName, one, two);
                    one.parent = parent;
                    two.parent = parent;
                    return parent;
                }
                else if (nary.Fun is UnaryOperator)
                {
                    Node one = CreateFromExpr(nary.Args[0]);
                    UnaryNode parent = new UnaryNode(nary.Fun.FunctionName, one);
                    one.parent = parent;
                    return parent;
                }
                else if (nary.Fun is FunctionCall)
                {
                    FunctionCall call = nary.Fun as FunctionCall;
                    if (nary.Args.Count == 1)
                    {
                        Node one = CreateFromExpr(nary.Args[0]);
                        UnaryNode parent = new UnaryNode(nary.Fun.FunctionName, one);
                        one.parent = parent;
                        return parent;
                    }
                    else if (nary.Args.Count == 2)
                    {
                        Node one = CreateFromExpr(nary.Args[0]);
                        Node two = CreateFromExpr(nary.Args[1]);
                        Node parent = new BinaryNode(call.FunctionName, one, two);
                        one.parent = parent;
                        two.parent = parent;
                        return parent;
                    }
                    else
                    {
                        Print.ExitMessage("Unhandled number of arguments in Boogie function call with function: " + nary.Fun.FunctionName);
                    }
                }
                else if (nary.Fun is MapSelect)
                {
                    List<Expr> indices = new List<Expr>();
                    while (true)
                    {
                        NAryExpr nary2 = nary.Args[0] as NAryExpr;
                        Print.ConditionalExitMessage(nary.Args.Count == 2, "Map select has more than two arguments");
                        indices.Insert(0, nary.Args[1]);
                        if (nary2 == null)
                            break;
                        else
                            nary = nary2;
                    }

                    IdentifierExpr identifier = nary.Args[0] as IdentifierExpr;
                    Node parent = new MapSymbolNode(identifier.Name);
                    foreach (Expr index in indices)
                    {
                        Node child = CreateFromExpr(index);
                        parent.Children.Add(child);
                        child.parent = parent;
                    }

                    return parent;
                }
                else
                {
                    Print.ExitMessage("Unhandled Nary expression: " + nary.Fun.GetType().ToString());
                }
            }
            else if (expr is IdentifierExpr)
            {
                IdentifierExpr identifier = expr as IdentifierExpr;
                return new ScalarSymbolNode(identifier.Name, identifier.Type);
            }
            else if (expr is LiteralExpr)
            {
                LiteralExpr literal = expr as LiteralExpr;
                if (literal.Val is BvConst)
                {
                    BvConst bv = (BvConst)literal.Val;
                    return new LiteralNode(new BitVector(bv));
                }
                else if (literal.Val is BigNum)
                {
                    BigNum num = (BigNum)literal.Val;
                    return new LiteralNode(new BitVector(num.ToInt));
                }
                else if (literal.Val is bool)
                {
                    bool boolean = (bool)literal.Val;
                    if (boolean)
                        return new LiteralNode(BitVector.True);
                    else
                        return new LiteralNode(BitVector.False);
                }
                else
                {
                    Print.ExitMessage("Unhandled literal expression: " + literal.ToString());
                }
            }
            else if (expr is BvExtractExpr)
            {
                BvExtractExpr bvExtract = expr as BvExtractExpr;
                Node child = CreateFromExpr(bvExtract.Bitvector);
                Node parent = new BVExtractNode(child, bvExtract.End, bvExtract.Start);
                child.parent = parent;
                return parent;
            }
            else if (expr is BvConcatExpr)
            {
                BvConcatExpr bvConcat = expr as BvConcatExpr;
                Node one = CreateFromExpr(bvConcat.E0);
                Node two = CreateFromExpr(bvConcat.E1);
                Node parent = new BVConcatenationNode(one, two);
                one.parent = parent;
                two.parent = parent;
                return parent;
            }
            else if (expr is ForallExpr)
            {
                ForallExpr forall = expr as ForallExpr;
                Node one = CreateFromExpr(forall._Body);
                Node parent = new ForAllNode(one);
                one.parent = parent;
                return parent;
            }

            Print.ExitMessage("Unhandled expression tree: " + expr.ToString() + " " + expr.Type.ToString());
            return null;
        }
    }

    public class ExprNode : Node
    {
        public BitVector Evaluation { get; set; } = null;

        public override void ClearState()
        {
            Evaluation = null;
            Initialised = true;
        }
    }

    public class OpNode : ExprNode
    {
        public string Op { get; private set; }

        public OpNode(string op)
            : base()
        {
            this.Op = op;
        }

        public override string ToString()
        {
            return Op;
        }
    }

    public class TernaryNode : OpNode
    {
        public TernaryNode(string op, Node one, Node two, Node three)
            : base(op)
        {
            Children.Add(one);
            Children.Add(two);
            Children.Add(three);
        }
    }

    public class BinaryNode : OpNode
    {
        public BinaryNode(string op, Node one, Node two)
            : base(op)
        {
            Children.Add(one);
            Children.Add(two);
        }
    }

    public class UnaryNode : OpNode
    {
        public UnaryNode(string op, Node one)
            : base(op)
        {
            Children.Add(one);
        }
    }

    public class ScalarSymbolNode : ExprNode
    {
        public string Symbol { get; private set; }

        public Microsoft.Boogie.Type Type { get; private set; }

        public bool IsOffsetVariable { get; private set; }

        public ScalarSymbolNode(string symbol, Microsoft.Boogie.Type type)
        {
            Symbol = symbol;
            Type = type;
            IsOffsetVariable = BoogieInterpreter.RegularExpressions.OffsetVariable.IsMatch(symbol);
        }

        public override string ToString()
        {
            return Symbol;
        }
    }

    public class MapSymbolNode : ExprNode
    {
        public string Basename { get; private set; }

        public MapSymbolNode(string basename)
        {
            this.Basename = basename;
        }

        public override string ToString()
        {
            return Basename;
        }
    }

    public class BVExtractNode : ExprNode
    {
        public int High { get; private set; }

        public int Low { get; private set; }

        public BVExtractNode(Node one, int high, int low)
        {
            Children.Add(one);
            High = high;
            Low = low;
        }

        public override string ToString()
        {
            return "[" + High.ToString() + ":" + Low.ToString() + "]";
        }
    }

    public class BVConcatenationNode : ExprNode
    {
        public BVConcatenationNode(Node one, Node two)
        {
            Children.Add(one);
            Children.Add(two);
        }
    }

    public class ForAllNode : ExprNode
    {
        public ForAllNode(Node one)
        {
            Children.Add(one);
        }
    }

    public class LiteralNode : ExprNode
    {
        public LiteralNode(BitVector val)
        {
            Evaluation = val;
        }

        public override string ToString()
        {
            return Evaluation.ToString();
        }
    }
}
