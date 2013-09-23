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
        {
          if (D is Implementation)
            {
              FindCallSites(D as Implementation);
            }
        }
    }

    private void FindCallSites(Implementation impl)
    {
      FindCallSites(impl.Blocks);
    }

    private void FindCallSites(List<Block> blocks)
    {
      foreach (Block b in blocks)
        {
          FindCallSites(b);
        }
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
      foreach(Procedure p in CallSites.Keys)
        {
          for (int i = 0; i < p.InParams.Count(); i++)
            {
              LiteralExpr literal = null;
              bool arbitrary = false;

              foreach (CallCmd callCmd in CallSites[p])
                {
                  if (callCmd.Ins[i] == null)
                    {
                      arbitrary = true;
                      break;
                    }

                  if (callCmd.Ins[i] is LiteralExpr)
                    {
                      LiteralExpr l = callCmd.Ins[i] as LiteralExpr;

                      if (literal == null)
                        {
                          literal = l;
                        }
                      else if (!l.Equals(literal))
                        {
                          arbitrary = true;
                          break;
                        }
                    }
                }

              if (literal != null && !arbitrary)
                {
                  Expr e;
                  e = new IdentifierExpr(Token.NoToken, p.InParams[i]);
                  e = Expr.Eq(e, literal);
                  p.Requires.Add(new Requires(false, e));
                }
            }
        }
    }
  }
}