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

    public enum SourceLanguage
    {
        OpenCL, CUDA
    }

    public class GVCommandLineOptions : CommandLineOptions
    {
        public bool OnlyIntraGroupRaceChecking { get; private set; } = false;

        public bool DebugGPUVerify { get; private set; } = false;

        // Dimensionality of block = BlockHighestDim + 1
        public int BlockHighestDim { get; private set; } = 2;

        // Dimensionality of grid = GridHighestDim + 1
        public int GridHighestDim { get; private set; } = 2;

        public SourceLanguage SourceLanguage { get; private set; } = SourceLanguage.OpenCL;

        public bool DisplayLoopAbstractions { get; private set; } = false;

        public GVCommandLineOptions()
            : base("GPUVerify", "GPUVerify kernel analyser")
        {
        }

        protected override bool ParseOption(string name, CommandLineParseState ps)
        {
            if (name == "sourceLanguage")
            {
                if (ps.ConfirmArgumentCount(1))
                {
                    if (ps.args[ps.i] == "cl")
                    {
                        SourceLanguage = SourceLanguage.OpenCL;
                    }
                    else if (ps.args[ps.i] == "cu")
                    {
                        SourceLanguage = SourceLanguage.CUDA;
                    }
                }

                return true;
            }

            if (name == "blockHighestDim")
            {
                int highestDim = BlockHighestDim;
                ps.GetNumericArgument(ref highestDim, 3);
                BlockHighestDim = highestDim;
                return true;
            }

            if (name == "gridHighestDim")
            {
                int highestDim = GridHighestDim;
                ps.GetNumericArgument(ref highestDim, 3);
                GridHighestDim = highestDim;
                return true;
            }

            if (name == "debugGPUVerify")
            {
                DebugGPUVerify = true;
                return true;
            }

            if (name == "onlyIntraGroupRaceChecking")
            {
                OnlyIntraGroupRaceChecking = true;
                return true;
            }

            if (name == "raceChecking")
            {
                if (ps.ConfirmArgumentCount(1))
                {
                    if (ps.args[ps.i] == "ORIGINAL")
                    {
                        RaceInstrumentationUtil.RaceCheckingMethod = RaceCheckingMethod.ORIGINAL;
                    }
                    else if (ps.args[ps.i] == "SINGLE")
                    {
                        RaceInstrumentationUtil.RaceCheckingMethod = RaceCheckingMethod.WATCHDOG_SINGLE;
                    }
                    else if (ps.args[ps.i] == "MULTIPLE")
                    {
                        RaceInstrumentationUtil.RaceCheckingMethod = RaceCheckingMethod.WATCHDOG_MULTIPLE;
                    }
                }

                return true;
            }

            if (name == "displayLoopAbstractions")
            {
                DisplayLoopAbstractions = true;
                return true;
            }

            return base.ParseOption(name, ps);  // defer to superclass
        }
    }
}
