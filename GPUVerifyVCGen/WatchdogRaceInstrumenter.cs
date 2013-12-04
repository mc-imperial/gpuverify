using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class WatchdogRaceInstrumenter : IRaceInstrumenter
  {

    public void AddRaceCheckingCandidateInvariants(Implementation impl, IRegion region) {
      throw new NotImplementedException();
    }

    public void AddKernelPrecondition() {
      throw new NotImplementedException();
    }

    public void AddRaceCheckingInstrumentation() {
      throw new NotImplementedException();
    }

    public void AddRaceCheckingDeclarations() {
      throw new NotImplementedException();
    }

    public BigBlock MakeResetReadWriteSetStatements(Variable v, Expr ResetCondition) {
      throw new NotImplementedException();
    }

    public void AddRaceCheckingCandidateRequires(Procedure Proc) {
      throw new NotImplementedException();
    }

    public void AddRaceCheckingCandidateEnsures(Procedure Proc) {
      throw new NotImplementedException();
    }
  }
}
