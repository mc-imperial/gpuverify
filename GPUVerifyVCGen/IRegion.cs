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
using Microsoft.Boogie;

namespace GPUVerify {

interface IRegion {
  object Identifier();
  IEnumerable<Cmd> Cmds();
  IEnumerable<object> CmdsChildRegions();
  IEnumerable<IRegion> SubRegions();
  IEnumerable<Block> PreHeaders();
  Expr Guard();
  void AddInvariant(PredicateCmd pc);
  List<PredicateCmd> RemoveInvariants();
}

}
