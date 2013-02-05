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

    public IKernelArrayInfo NonLocalStateToCheck;

    public void setVerifier(GPUVerifier verifier) {
      this.verifier = verifier;
      NonLocalStateToCheck = new KernelArrayInfoLists();
      foreach (Variable v in verifier.KernelArrayInfo.getGlobalArrays()) {
        NonLocalStateToCheck.getGlobalArrays().Add(v);
      }
      foreach (Variable v in verifier.KernelArrayInfo.getGroupSharedArrays()) {
        NonLocalStateToCheck.getGroupSharedArrays().Add(v);
      }
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

    private CmdSeq AddNoAccessAssumes(CmdSeq cs) {
      var result = new CmdSeq();
      foreach (Cmd c in cs) {
        result.Add(c);
        if (c is AssignCmd) {
          AssignCmd assign = c as AssignCmd;

          ReadCollector rc = new ReadCollector(NonLocalStateToCheck);
          foreach (var rhs in assign.Rhss)
            rc.Visit(rhs);
          if (rc.accesses.Count > 0) {
            foreach (AccessRecord ar in rc.accesses) {
              AddNoAccessAssumes(result, ar);
            }
          }

          foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss)) {
            WriteCollector wc = new WriteCollector(NonLocalStateToCheck);
            wc.Visit(LhsRhs.Item1);
            if (wc.GetAccess() != null) {
              AccessRecord ar = wc.GetAccess();
              AddNoAccessAssumes(result, ar);
            }
          }
        }
      }
      return result;
    }

    private void AddNoAccessAssumes(CmdSeq result, AccessRecord ar) {
    //Causes System.InvalidOperationException: Collection was modified; enumeration operation may not execute.
    //result.Add(new AssumeCmd(Token.NoToken, Expr.Neq(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateNotAccessedVariable(ar.v.Name, ar.Index.Type)), ar.Index)));
      result.Add(new AssumeCmd(Token.NoToken, Expr.Neq(new IdentifierExpr(Token.NoToken, GPUVerifier.MakeNotAccessedVariable(ar.v.Name, ar.Index.Type)), ar.Index)));
    }

    private StmtList AddNoAccessAssumes(StmtList stmtList) {
      Contract.Requires(stmtList != null);

      StmtList result = new StmtList(new List<BigBlock>(), stmtList.EndCurly);

      foreach (BigBlock bodyBlock in stmtList.BigBlocks) {
        result.BigBlocks.Add(AddNoAccessAssumes(bodyBlock));
      }
      return result;
    }

    private BigBlock AddNoAccessAssumes(BigBlock bb) {
      BigBlock result = new BigBlock(bb.tok, bb.LabelName, AddNoAccessAssumes(bb.simpleCmds), null, bb.tc);

      if (bb.ec is WhileCmd) {
        WhileCmd WhileCommand = bb.ec as WhileCmd;
        result.ec = new WhileCmd(WhileCommand.tok, WhileCommand.Guard,
                WhileCommand.Invariants, AddNoAccessAssumes(WhileCommand.Body));
      }
      else if (bb.ec is IfCmd) {
        IfCmd IfCommand = bb.ec as IfCmd;
        Debug.Assert(IfCommand.elseIf == null); // We don't handle else if yet
        result.ec = new IfCmd(IfCommand.tok, IfCommand.Guard, AddNoAccessAssumes(IfCommand.thn), IfCommand.elseIf, IfCommand.elseBlock != null ? AddNoAccessAssumes(IfCommand.elseBlock) : null);
      }
      else if (bb.ec is BreakCmd) {
        result.ec = bb.ec;
      }
      else {
        Debug.Assert(bb.ec == null);
      }

      return result;
    }

  }

}
