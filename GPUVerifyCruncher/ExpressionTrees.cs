//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using System.Collections.Specialized;
using Microsoft.Basetypes;
using Microsoft.Boogie;

namespace GPUVerify
{
    public class ExprTree : System.Collections.IEnumerable
    {
        protected Dictionary<int, HashSet<Node>> levels = new Dictionary<int, HashSet<Node>>();
        protected List<Node> nodes = new List<Node>();
        protected Node root = null;
        protected int height = 0;
        public BitVector evaluation;
        public bool unitialised = false;
        public Expr expr;

        public ExprTree(Expr expr)
        {
            this.expr = expr;
            root = Node.CreateFromExpr(expr);
            levels[0] = new HashSet<Node>();
            levels[0].Add(root);
            SetLevels(root, 0);
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
                if (!(node is LiteralNode<BitVector>))
                    node.ClearState();
            }
            unitialised = false;
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
                builder.Append(String.Format("Level {0}", i)).Append(Environment.NewLine).Append(" : ");
                foreach (Node node in levels[i])
                {
                    builder.Append(node.ToString()).Append(" : ");
                }
                builder.Append(Environment.NewLine);
            }
            return builder.ToString();
        }
    }

    public abstract class Node
    {
        protected List<Node> children = new List<Node>();
        protected Node parent = null;
        public bool uninitialised = false;

        public Node()
        {
        }

        public bool IsLeaf()
        {
            return children.Count == 0;
        }

        public List<Node> GetChildren()
        {
            return children;
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
                    Node parent = new TernaryNode<BitVector>(nary.Fun.FunctionName, one, two, three);
                    one.parent = parent;
                    two.parent = parent;
                    three.parent = parent;
                    return parent;
                }
                else if (nary.Fun is BinaryOperator)
                {
                    Node one = CreateFromExpr(nary.Args[0]);
                    Node two = CreateFromExpr(nary.Args[1]);
                    Node parent = new BinaryNode<BitVector>(nary.Fun.FunctionName, one, two);
                    one.parent = parent;
                    two.parent = parent;
                    return parent;
                }
                else if (nary.Fun is UnaryOperator)
                {
                    Node one = CreateFromExpr(nary.Args[0]);
                    UnaryNode<BitVector> parent = new UnaryNode<BitVector>(nary.Fun.FunctionName, one);
                    one.parent = parent;
                    return parent;
                }
                else if (nary.Fun is FunctionCall)
                {
                    FunctionCall call = nary.Fun as FunctionCall;
                    if (nary.Args.Count == 1)
                    {
                        Node one = CreateFromExpr(nary.Args[0]);
                        UnaryNode<BitVector> parent = new UnaryNode<BitVector>(nary.Fun.FunctionName, one);
                        one.parent = parent;
                        return parent;
                    }
                    else if (nary.Args.Count == 2)
                    {
                        Node one = CreateFromExpr(nary.Args[0]);
                        Node two = CreateFromExpr(nary.Args[1]);
                        Node parent = new BinaryNode<BitVector>(call.FunctionName, one, two);
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
                    Node parent = new MapSymbolNode<BitVector>(identifier.Name);
                    foreach (Expr index in indices)
                    {
                        Node child = CreateFromExpr(index);
                        parent.children.Add(child);
                        child.parent = parent;
                    }
                    return parent;
                }
                else
                    Print.ExitMessage("Unhandled Nary expression: " + nary.Fun.GetType().ToString());
            }
            else if (expr is IdentifierExpr)
            {
                IdentifierExpr identifier = expr as IdentifierExpr;
                return new ScalarSymbolNode<BitVector>(identifier.Name);
            }
            else if (expr is LiteralExpr)
            {
                LiteralExpr literal = expr as LiteralExpr;
                if (literal.Val is BvConst)
                {
                    BvConst bv = (BvConst)literal.Val;
                    return new LiteralNode<BitVector>(new BitVector(bv));
                }
                else if (literal.Val is BigNum)
                {
                    BigNum num = (BigNum)literal.Val;
                    return new LiteralNode<BitVector>(new BitVector(num.ToInt));
                }
                else if (literal.Val is bool)
                {
                    bool boolean = (bool)literal.Val;
                    if (boolean)
                        return new LiteralNode<BitVector>(BitVector.True);
                    else
                        return new LiteralNode<BitVector>(BitVector.False);
                }
                else
                    Print.ExitMessage("Unhandled literal expression: " + literal.ToString());
            }
            else if (expr is BvExtractExpr)
            {
                BvExtractExpr bvExtract = expr as BvExtractExpr;
                Node child = CreateFromExpr(bvExtract.Bitvector);
                Node parent = new BVExtractNode<BitVector>(child, bvExtract.End, bvExtract.Start);
                child.parent = parent;
                return parent;
            }
			else if (expr is BvConcatExpr)
            {
                BvConcatExpr bvConcat = expr as BvConcatExpr;
                Node one = CreateFromExpr(bvConcat.E0);
                Node two = CreateFromExpr(bvConcat.E1);
                Node parent = new BVConcatenationNode<BitVector>(one, two);
                one.parent = parent;
                two.parent = parent;
                return parent;
            }
            
            Print.ExitMessage("Unhandled expression tree: " + expr.ToString() + " " + expr.Type.ToString());
            return null;
        }
    }

    public class ExprNode<T> : Node
    {
        public HashSet<T> evaluations = new HashSet<T>();

        public override void ClearState()
        {
            evaluations.Clear();
            uninitialised = false;
        }
        
        public T GetUniqueElement ()
        {
            if (evaluations.Count != 1)
                throw new UnhandledException("There is no unique element in the evaluation set");
            return evaluations.First();
        }
    }

    public class OpNode<T> : ExprNode<T>
    {
        public string op;

        public OpNode(string op):
			base()
        {
            this.op = op;
        }

        public override string ToString()
        {
            return op + " (" + typeof(T).ToString() + ")";
        }
    }

    public class TernaryNode<T> : OpNode<T>
    {
        public TernaryNode(string op, Node one, Node two, Node three):
			base(op)
        {
            children.Add(one);
            children.Add(two);
            children.Add(three);
        }
    }

    public class BinaryNode<T> : OpNode<T>
    {
        public BinaryNode(string op, Node one, Node two):
			base(op)
        {
            children.Add(one);
            children.Add(two);
        }
    }

    public class UnaryNode<T> : OpNode<T>
    {
        public UnaryNode(string op, Node one):
			base(op)
        {
            children.Add(one);
        }
    }

    public class ScalarSymbolNode<T> : ExprNode<T>
    {
        public string symbol;

        public ScalarSymbolNode(string symbol)
        {
            this.symbol = symbol;
        }

        public override string ToString()
        {
            return symbol;
        }
    }

    public class MapSymbolNode<T> : ExprNode<T>
    {
        public string basename;

        public MapSymbolNode(string basename)
        {
            this.basename = basename;
        }

        public override string ToString()
        {
            return basename;
        }
    }
    
    public class BVExtractNode<T> : ExprNode<T>
    {
        public int low;
        public int high;

        public BVExtractNode(Node one, int high, int low)
        {
            children.Add(one);
            this.high = high;
            this.low = low;
        }

        public override string ToString()
        {
            return "[" + high.ToString() + ":" + low.ToString() + "]";
        }
    }
    
    public class BVConcatenationNode<T> : ExprNode<T>
    {
        public BVConcatenationNode(Node one, Node two)
        {
            children.Add(one);
            children.Add(two);
        }
    }
    

    public class LiteralNode<T> : ExprNode<T>
    {
        public LiteralNode(T val)
        {
            evaluations.Add(val);
        }

        public override string ToString()
        {
            return GetUniqueElement().ToString();
        }
    }
}

