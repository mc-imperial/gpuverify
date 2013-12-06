using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class WatchdogSingleRaceInstrumenter : RaceInstrumenter
  {
    internal WatchdogSingleRaceInstrumenter(GPUVerifier verifier) : base(verifier) {

    }

    protected override void AddLogAccessProcedure(Variable v, AccessType Access) {
      throw new NotImplementedException();
    }

    protected override void AddCheckAccessProcedure(Variable v, AccessType Access) {
      throw new NotImplementedException();
    }

  }
}
