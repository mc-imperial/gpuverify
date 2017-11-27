//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System.Collections.Generic;
using Microsoft.Boogie;

namespace GPUVerify {

interface IRegion {
  object Identifier();
  IEnumerable<Cmd> Cmds();
  IEnumerable<object> CmdsChildRegions();
  IEnumerable<IRegion> SubRegions();
  IEnumerable<Block> PreHeaders();
  Block Header();
  IEnumerable<Block> SubBlocks();
  Expr Guard();
  void AddInvariant(PredicateCmd pc);
  void AddLoopInvariantDisabledTag();
  List<PredicateCmd> RemoveInvariants();
  HashSet<Variable> PartitionVariablesOfHeader();
  HashSet<Variable> PartitionVariablesOfRegion();
}

}
