//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify
{
    using Microsoft.Boogie;

    public enum RaceCheckingMethod
    {
        ORIGINAL, WATCHDOG_SINGLE, WATCHDOG_MULTIPLE
    }

    public class RaceInstrumentationUtil
    {
        // Assigned by GVCommandLineOptions
        public static RaceCheckingMethod RaceCheckingMethod { get; set; } = RaceCheckingMethod.WATCHDOG_SINGLE;

        public static string MakeOffsetVariableName(string name, AccessType access)
        {
            if (RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_SINGLE)
            {
                return "_WATCHED_OFFSET";
            }

            if (RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_MULTIPLE)
            {
                return "_WATCHED_OFFSET_" + name;
            }

            return "_" + access + "_OFFSET_" + name;
        }

        public static string MakeHasOccurredVariableName(string name, AccessType access)
        {
            return "_" + access + "_HAS_OCCURRED_" + name;
        }

        public static Variable MakeOffsetVariable(string name, AccessType access, Type type)
        {
            var ident = new TypedIdent(Token.NoToken, MakeOffsetVariableName(name, access), type);
            if (RaceCheckingMethod == RaceCheckingMethod.ORIGINAL)
            {
                return new GlobalVariable(Token.NoToken, ident);
            }

            return new Constant(Token.NoToken, ident, false);
        }

        public static string MakeValueVariableName(string name, AccessType access)
        {
            if (RaceCheckingMethod == RaceCheckingMethod.ORIGINAL)
            {
                return "_" + access + "_VALUE_" + name;
            }

            return "_WATCHED_VALUE_" + name;
        }

        public static Variable MakeValueVariable(string name, AccessType access, Type type)
        {
            var ident = new TypedIdent(Token.NoToken, MakeValueVariableName(name, access), type);
            if (RaceCheckingMethod == RaceCheckingMethod.ORIGINAL)
            {
                return new GlobalVariable(Token.NoToken, ident);
            }

            return new Constant(Token.NoToken, ident, false);
        }

        public static string MakeAsyncHandleVariableName(string name, AccessType access)
        {
            return "_" + access + "_ASYNC_HANDLE_" + name;
        }

        public static Variable MakeAsyncHandleVariable(string name, AccessType access, Type type)
        {
            var ident = new TypedIdent(Token.NoToken, MakeAsyncHandleVariableName(name, access), type);
            return new GlobalVariable(Token.NoToken, ident);
        }
    }
}
