using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class WatchdogMultipleRaceInstrumenter : RaceInstrumenter
  {
    internal WatchdogMultipleRaceInstrumenter(GPUVerifier verifier) : base(verifier) {

    }

    protected override void AddLogAccessProcedure(Variable v, AccessType Access) {
      throw new NotImplementedException();
    }

  }
}
