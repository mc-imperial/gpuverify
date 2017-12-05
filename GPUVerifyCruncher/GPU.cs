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
    using System;
    using System.Collections.Generic;
    using System.Text;
    using System.Text.RegularExpressions;

    public enum DIMENSION
    {
        X, Y, Z
    }

    public class GPU
    {
        public Dictionary<DIMENSION, int> GridDim { get; private set; } = new Dictionary<DIMENSION, int>()
            { { DIMENSION.X, -1 }, { DIMENSION.Y, -1 }, { DIMENSION.Z, -1 } };

        public Dictionary<DIMENSION, int> BlockDim { get; private set; } = new Dictionary<DIMENSION, int>()
            { { DIMENSION.X, -1 }, { DIMENSION.Y, -1 }, { DIMENSION.Z, -1 } };

        public Dictionary<DIMENSION, int> GridOffset { get; private set; } = new Dictionary<DIMENSION, int>()
            { { DIMENSION.X, -1 }, { DIMENSION.Y, -1 }, { DIMENSION.Z, -1 } };

        public GPU()
        {
        }

        public void SetBlockDim(Tuple<int, int, int> blockDim)
        {
            this.BlockDim[DIMENSION.X] = blockDim.Item1;
            this.BlockDim[DIMENSION.Y] = blockDim.Item2;
            this.BlockDim[DIMENSION.Z] = blockDim.Item3;
        }

        public void SetGridDim(Tuple<int, int, int> gridDim)
        {
            this.GridDim[DIMENSION.X] = gridDim.Item1;
            this.GridDim[DIMENSION.Y] = gridDim.Item2;
            this.GridDim[DIMENSION.Z] = gridDim.Item3;
        }

        public void SetGridOffset(Tuple<int, int, int> gridOffset)
        {
            this.GridOffset[DIMENSION.X] = gridOffset.Item1;
            this.GridOffset[DIMENSION.Y] = gridOffset.Item2;
            this.GridOffset[DIMENSION.Z] = gridOffset.Item3;
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            builder.Append(string.Format("blockDim=[{0},{1},{2}]", BlockDim[DIMENSION.X], BlockDim[DIMENSION.Y], BlockDim[DIMENSION.Z]));
            builder.Append("\n");
            builder.Append(string.Format("gridDim =[{0},{1},{2}]", GridDim[DIMENSION.X], GridDim[DIMENSION.Y], GridDim[DIMENSION.Z]));
            builder.Append(string.Format("gridOffset =[{0},{1},{2}]", GridOffset[DIMENSION.X], GridOffset[DIMENSION.Y], GridOffset[DIMENSION.Z]));
            return builder.ToString();
        }

        public static bool IsLocalIDName(string name)
        {
            return Regex.IsMatch(name, "local_id_[x|y|z]", RegexOptions.IgnoreCase);
        }

        public static bool IsThreadBlockSizeName(string name)
        {
            return Regex.IsMatch(name, "group_size_[x|y|z]", RegexOptions.IgnoreCase);
        }

        public static bool IsGridBlockSizeName(string name)
        {
            return Regex.IsMatch(name, "num_groups_[x|y|z]", RegexOptions.IgnoreCase);
        }

        public static bool IsGridOffsetName(string name)
        {
            return Regex.IsMatch(name, "global_offset_[x|y|z]", RegexOptions.IgnoreCase);
        }
    }
}
