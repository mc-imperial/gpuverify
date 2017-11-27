//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using Microsoft.Boogie;

namespace GPUVerify
{

  public enum RaceCheckingMethod {
    ORIGINAL, WATCHDOG_SINGLE, WATCHDOG_MULTIPLE
  }

  public class RaceInstrumentationUtil
  {

    public static RaceCheckingMethod RaceCheckingMethod = RaceCheckingMethod.WATCHDOG_SINGLE;

    public static string MakeOffsetVariableName(string Name, AccessType Access)
    {
      if(RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_SINGLE) {
        return "_WATCHED_OFFSET";
      }
      if(RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_MULTIPLE) {
        return "_WATCHED_OFFSET_" + Name;
      }
      return "_" + Access + "_OFFSET_" + Name;
    }

    public static string MakeHasOccurredVariableName(string Name, AccessType Access)
    {
        return "_" + Access + "_HAS_OCCURRED_" + Name;
    }

    public static Variable MakeOffsetVariable(string Name, AccessType Access, Microsoft.Boogie.Type Type)
    {
      var Ident = new TypedIdent(Token.NoToken, MakeOffsetVariableName(Name, Access),
          Type);
      if(RaceCheckingMethod == RaceCheckingMethod.ORIGINAL) {
        return new GlobalVariable(Token.NoToken, Ident);
      }
      return new Constant(Token.NoToken, Ident, false);
    }

    public static string MakeValueVariableName(string Name, AccessType Access) {
      if(RaceCheckingMethod == RaceCheckingMethod.ORIGINAL) {
        return "_" + Access + "_VALUE_" + Name;
      }
      return "_WATCHED_VALUE_" + Name;
    }

    public static Variable MakeValueVariable(string Name, AccessType Access, Microsoft.Boogie.Type Type) {
      var Ident = new TypedIdent(Token.NoToken, MakeValueVariableName(Name, Access),
          Type);
      if(RaceCheckingMethod == RaceCheckingMethod.ORIGINAL) {
        return new GlobalVariable(Token.NoToken, Ident);
      }
      return new Constant(Token.NoToken, Ident, false);
    }

    public static string MakeAsyncHandleVariableName(string Name, AccessType Access)
    {
        return "_" + Access + "_ASYNC_HANDLE_" + Name;
    }

    public static Variable MakeAsyncHandleVariable(string Name, AccessType Access, Microsoft.Boogie.Type Type)
    {
      var Ident = new TypedIdent(Token.NoToken, MakeAsyncHandleVariableName(Name, Access),
          Type);
      return new GlobalVariable(Token.NoToken, Ident);
    }

  }
}
