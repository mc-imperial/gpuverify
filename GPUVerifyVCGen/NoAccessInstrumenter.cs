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
using System.IO;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using Microsoft.Boogie;
using Microsoft.Basetypes;

namespace GPUVerify {

  class NoAccessInstrumenter : INoAccessInstrumenter {

    protected GPUVerifier verifier;

    public IKernelArrayInfo StateToCheck;

    public NoAccessInstrumenter(GPUVerifier verifier) {
      this.verifier = verifier;
      StateToCheck = verifier.KernelArrayInfo;
    }

    public void AddNoAccessInstrumentation() {
      foreach (Declaration d in verifier.Program.TopLevelDeclarations) {
        if (d is Implementation) {
          AddNoAccessAssumes(d as Implementation);
        }
      }
    }

    private void AddNoAccessAssumes(Implementation impl) {
      impl.Blocks = impl.Blocks.Select(AddNoAccessAssumes).ToList();
    }

    private Block AddNoAccessAssumes(Block b) {
      b.Cmds = AddNoAccessAssumes(b.Cmds);
      return b;
    }

    private List<Cmd> AddNoAccessAssumes(List<Cmd> cs) {
      var result = new List<Cmd>();
      foreach (Cmd c in cs) {
        result.Add(c);
        if (c is AssignCmd) {
          AssignCmd assign = c as AssignCmd;

          ReadCollector rc = new ReadCollector(StateToCheck);
          foreach (var rhs in assign.Rhss)
            rc.Visit(rhs);
          if (rc.accesses.Count > 0) {
            foreach (AccessRecord ar in rc.accesses) {
              AddNoAccessAssumes(result, ar);
            }
          }

          foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss)) {
            WriteCollector wc = new WriteCollector(StateToCheck);
            wc.Visit(LhsRhs.Item1);
            if (wc.FoundWrite()) {
              AccessRecord ar = wc.GetAccess();
              AddNoAccessAssumes(result, ar);
            }
          }
        }
      }
      return result;
    }

    private void AddNoAccessAssumes(List<Cmd> result, AccessRecord ar) {
    // Revisit: Following causes System.InvalidOperationException: Collection was modified; enumeration operation may not execute.
    //result.Add(new AssumeCmd(Token.NoToken, Expr.Neq(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateNotAccessedVariable(ar.v.Name, ar.Index.Type)), ar.Index)));
      result.Add(new AssumeCmd(Token.NoToken, Expr.Neq(new IdentifierExpr(Token.NoToken, GPUVerifier.MakeNotAccessedVariable(ar.v.Name, ar.Index.Type)), ar.Index)));
    }

  }

}
