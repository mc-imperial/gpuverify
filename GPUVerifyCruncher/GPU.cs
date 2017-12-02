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

    public enum DIMENSION { X, Y, Z }

    public class GPU
    {
        public Dictionary<DIMENSION, int> gridDim = new Dictionary<DIMENSION, int>();
        public Dictionary<DIMENSION, int> blockDim = new Dictionary<DIMENSION, int>();
        public Dictionary<DIMENSION, int> gridOffset = new Dictionary<DIMENSION, int>();

        public GPU()
        {
            gridDim[DIMENSION.X] = -1;
            gridDim[DIMENSION.Y] = -1;
            gridDim[DIMENSION.Z] = -1;
            blockDim[DIMENSION.X] = -1;
            blockDim[DIMENSION.Y] = -1;
            blockDim[DIMENSION.Z] = -1;
            gridOffset[DIMENSION.X] = -1;
            gridOffset[DIMENSION.Y] = -1;
            gridOffset[DIMENSION.Z] = -1;
        }

        public void SetBlockDim(Tuple<int, int, int> blockDim)
        {
            this.blockDim[DIMENSION.X] = blockDim.Item1;
            this.blockDim[DIMENSION.Y] = blockDim.Item2;
            this.blockDim[DIMENSION.Z] = blockDim.Item3;
        }

        public void SetGridDim(Tuple<int, int, int> gridDim)
        {
            this.gridDim[DIMENSION.X] = gridDim.Item1;
            this.gridDim[DIMENSION.Y] = gridDim.Item2;
            this.gridDim[DIMENSION.Z] = gridDim.Item3;
        }

        public void SetGridOffset(Tuple<int, int, int> gridOffset)
        {
            this.gridOffset[DIMENSION.X] = gridOffset.Item1;
            this.gridOffset[DIMENSION.Y] = gridOffset.Item2;
            this.gridOffset[DIMENSION.Z] = gridOffset.Item3;
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            builder.Append(string.Format("blockDim=[{0},{1},{2}]", blockDim[DIMENSION.X], blockDim[DIMENSION.Y], blockDim[DIMENSION.Z]));
            builder.Append("\n");
            builder.Append(string.Format("gridDim =[{0},{1},{2}]", gridDim[DIMENSION.X], gridDim[DIMENSION.Y], gridDim[DIMENSION.Z]));
            builder.Append(string.Format("gridOffset =[{0},{1},{2}]", gridOffset[DIMENSION.X], gridOffset[DIMENSION.Y], gridOffset[DIMENSION.Z]));
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
