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

  class ConstantWriteInstrumenter : IConstantWriteInstrumenter {

    protected GPUVerifier verifier;

    private QKeyValue SourceLocationAttributes = null;

    public IKernelArrayInfo StateToCheck;

    public ConstantWriteInstrumenter(GPUVerifier verifier) {
      this.verifier = verifier;
      StateToCheck = verifier.KernelArrayInfo;
    }

    public void AddConstantWriteInstrumentation() {
      foreach (Declaration d in verifier.Program.TopLevelDeclarations) {
        if (d is Implementation) {
          AddConstantWriteAsserts(d as Implementation);
        }
      }
    }

    private void AddConstantWriteAsserts(Implementation impl) {
      impl.Blocks = impl.Blocks.Select(AddConstantWriteAsserts).ToList();
    }

    private Block AddConstantWriteAsserts(Block b) {
      b.Cmds = AddConstantWriteAsserts(b.Cmds);
      return b;
    }

    private List<Cmd> AddConstantWriteAsserts(List<Cmd> cs) {
      var result = new List<Cmd>();
      foreach (Cmd c in cs) {
        result.Add(c);

        if (c is AssertCmd) {
          AssertCmd assertion = c as AssertCmd;
          if (QKeyValue.FindBoolAttribute(assertion.Attributes, "sourceloc")) {
            SourceLocationAttributes = assertion.Attributes;
            // Do not remove source location assertions
            // This is done by the race instrumenter
          }
        }

        if (c is AssignCmd) {
          AssignCmd assign = c as AssignCmd;

          foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss)) {
            ConstantWriteCollector cwc = new ConstantWriteCollector(StateToCheck);
            cwc.Visit(LhsRhs.Item1);
            if (cwc.FoundWrite()) {
              AssertCmd constantAssert = new AssertCmd(Token.NoToken, Expr.False);
              constantAssert.Attributes = SourceLocationAttributes;
              constantAssert.Attributes
                  = new QKeyValue(Token.NoToken, "constant_write", new List<object>(), constantAssert.Attributes);
              result.Add(constantAssert);
            }
          }
        }
      }
      return result;
    }

    private StmtList AddConstantWriteAsserts(StmtList stmtList) {
      Contract.Requires(stmtList != null);

      StmtList result = new StmtList(new List<BigBlock>(), stmtList.EndCurly);

      foreach (BigBlock bodyBlock in stmtList.BigBlocks) {
        result.BigBlocks.Add(AddConstantWriteAsserts(bodyBlock));
      }
      return result;
    }

    private BigBlock AddConstantWriteAsserts(BigBlock bb) {
      BigBlock result = new BigBlock(bb.tok, bb.LabelName, AddConstantWriteAsserts(bb.simpleCmds), null, bb.tc);

      if (bb.ec is WhileCmd) {
        WhileCmd WhileCommand = bb.ec as WhileCmd;
        result.ec = new WhileCmd(WhileCommand.tok, WhileCommand.Guard,
                WhileCommand.Invariants, AddConstantWriteAsserts(WhileCommand.Body));
      }
      else if (bb.ec is IfCmd) {
        IfCmd IfCommand = bb.ec as IfCmd;
        Debug.Assert(IfCommand.elseIf == null); // We don't handle else if yet
        result.ec = new IfCmd(IfCommand.tok, IfCommand.Guard, AddConstantWriteAsserts(IfCommand.thn), IfCommand.elseIf, IfCommand.elseBlock != null ? AddConstantWriteAsserts(IfCommand.elseBlock) : null);
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
