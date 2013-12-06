using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GPUVerify
{
  public class RaceInstrumentationUtil
  {

    public static string MakeOffsetVariableName(string Name, AccessType Access)
    {
        return "_" + Access + "_OFFSET_" + Name;
    }

    public static string MakeHasOccurredVariableName(string Name, AccessType Access)
    {
        return "_" + Access + "_HAS_OCCURRED_" + Name;
    }

  }
}
