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
  class CallSiteAnalyser
  {
    private GPUVerifier verifier;
    private Dictionary<Procedure, List<CallCmd>> CallSites;

    public CallSiteAnalyser(GPUVerifier verifier)
    {
      this.verifier = verifier;
      CallSites = new Dictionary<Procedure, List<CallCmd>>();
    }

    internal void Analyse()
    {
      FindAllCallSites();
      LiteralArgumentAnalyser();
    }

    private void FindAllCallSites()
    {
      foreach (Declaration D in verifier.Program.TopLevelDeclarations)
        if (D is Implementation)
          FindCallSites(D as Implementation);
    }

    private void FindCallSites(Implementation impl)
    {
      FindCallSites(impl.Blocks);
    }

    private void FindCallSites(List<Block> blocks)
    {
      foreach (Block b in blocks)
        FindCallSites(b);
    }

    private void FindCallSites(Block b)
    {
      FindCallSites(b.Cmds);
    }

    private void FindCallSites(List<Cmd> cs)
    {
      foreach (Cmd c in cs)
        {
          if (c is CallCmd)
            {
              CallCmd callCmd = c as CallCmd;

              // Only consider non-inlined procedures
              if (QKeyValue.FindIntAttribute(callCmd.Proc.Attributes,
                                             "inline", -1) != -1)
                continue;

              if (!CallSites.ContainsKey(callCmd.Proc))
                {
                  CallSites[callCmd.Proc] = new List<CallCmd>();
                }

              CallSites[callCmd.Proc].Add(callCmd);
            }
        }
    }

    private void LiteralArgumentAnalyser()
    {
      foreach (Procedure p in CallSites.Keys)
        for (int i = 0; i < p.InParams.Count(); i++)
          LiteralArgumentAnalyser(p, i);
    }

    private void LiteralArgumentAnalyser(Procedure p, int arg)
    {
      LiteralExpr literal = null;

      foreach (CallCmd callCmd in CallSites[p])
        {
          if (callCmd.Ins[arg] == null)
            return;

          if (!(callCmd.Ins[arg] is LiteralExpr))
            continue;

          LiteralExpr l = callCmd.Ins[arg] as LiteralExpr;

          if (literal == null)
            literal = l;
          else if (!literal.Equals(l))
            return;

          Expr e;
          e = new IdentifierExpr(Token.NoToken, p.InParams[arg]);
          e = Expr.Eq(e, literal);
          p.Requires.Add(new Requires(false, e));
        }
    }
  }
}
