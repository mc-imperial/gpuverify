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
  public enum ToolExitCodes : int
  {
    SUCCESS = 0,
    // This is a hack so that raised exceptions always have the same error code.
    // If we have uncaught exceptions (i.e. with -DebugGPUVerify) then mono will exit with this exit code 1.
    // Not sure what happens on Windows.
    INTERNAL_ERROR = 1,
    // Uncategorised failure probably due to external input (e.g. invalid input program)
    OTHER_ERROR = 2,
    // A bug in the input Boogie program will be reported
    VERIFICATION_ERROR = 3,
  }
}
