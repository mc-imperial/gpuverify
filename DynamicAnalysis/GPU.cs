using System;
using System.Text;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace DynamicAnalysis
{
	public enum DIMENSION {X, Y, Z};
	
	public class GPU
	{
		private static GPU instance;
		public Dictionary<DIMENSION, int> gridDim  = new Dictionary<DIMENSION, int>();
		public Dictionary<DIMENSION, int> blockDim = new Dictionary<DIMENSION, int>();
		public Dictionary<DIMENSION, int> threadID = new Dictionary<DIMENSION, int>();
		public Dictionary<DIMENSION, int> groupID  = new Dictionary<DIMENSION, int>();
		
		private GPU ()
		{
			gridDim[DIMENSION.X]  = -1;
			gridDim[DIMENSION.Y]  = -1;
			gridDim[DIMENSION.Z]  = -1;
			blockDim[DIMENSION.X] = -1;
			blockDim[DIMENSION.Y] = -1;
			blockDim[DIMENSION.Z] = -1;
			threadID[DIMENSION.X] = -1;
			threadID[DIMENSION.Y] = -1;
			threadID[DIMENSION.Z] = -1;
			groupID[DIMENSION.X]  = -1;
			groupID[DIMENSION.Y]  = -1;
			groupID[DIMENSION.Z]  = -1;
		}
		
		public static GPU Instance
		{
			get
			{
				if (instance == null)
					instance = new GPU();
				return instance;
			}
		}
		
		public void SetBlockDim (Tuple<int, int, int> blockDim)
		{
			this.blockDim[DIMENSION.X] = blockDim.Item1;
			this.blockDim[DIMENSION.Y] = blockDim.Item2;
			this.blockDim[DIMENSION.Z] = blockDim.Item3;
		}
		
		public void SetGridDim (Tuple<int, int, int> gridDim)
		{
			this.gridDim[DIMENSION.X] = gridDim.Item1;
			this.gridDim[DIMENSION.Y] = gridDim.Item2;
			this.gridDim[DIMENSION.Z] = gridDim.Item3;
		}
		
		public void SetThreadID (Tuple<int, int, int> threadID)
		{
			this.threadID[DIMENSION.X] = threadID.Item1;
			this.threadID[DIMENSION.Y] = threadID.Item2;
			this.threadID[DIMENSION.Z] = threadID.Item3;
		}
		
		public void SetGroupID (Tuple<int, int, int> groupID)
		{
			this.groupID[DIMENSION.X] = groupID.Item1;
			this.groupID[DIMENSION.Y] = groupID.Item2;
			this.groupID[DIMENSION.Z] = groupID.Item3;
		}
		
		public bool IsUserSetThreadID (DIMENSION dim)
		{
			return threadID[dim] != -1;
		}
		
		public bool IsUserSetGroupID (DIMENSION dim)
		{
			return groupID[dim] != -1;
		}
		
		public bool IsUserSetBlockDim (DIMENSION dim)
		{
			return blockDim[dim] != -1;
		}
		
		public bool IsUserSetGridDim (DIMENSION dim)
		{
			return gridDim[dim] != -1;
		}
		
		public override string ToString()
		{
			StringBuilder builder = new StringBuilder();
			builder.Append(String.Format("blockDim=[{0},{1},{2}]", blockDim[DIMENSION.X], blockDim[DIMENSION.Y], blockDim[DIMENSION.Z]));
			builder.Append("\n");
			builder.Append(String.Format("gridDim =[{0},{1},{2}]", gridDim[DIMENSION.X], gridDim[DIMENSION.Y], gridDim[DIMENSION.Z]));
			builder.Append("\n");
			builder.Append(String.Format("threadID=[{0},{1},{2}]", threadID[DIMENSION.X], threadID[DIMENSION.Y], threadID[DIMENSION.Z]));
			builder.Append("\n");
			builder.Append(String.Format("groupID =[{0},{1},{2}]", groupID[DIMENSION.X], groupID[DIMENSION.Y], groupID[DIMENSION.Z]));
			return builder.ToString();
		}
		
		public static bool IsLocalIDName (string name)
		{
			return Regex.IsMatch(name, "local_id_[x|y|z]", RegexOptions.IgnoreCase);
		}
		
		public static bool IsThreadBlockSizeName (string name)
		{
			return Regex.IsMatch(name, "group_size_[x|y|z]", RegexOptions.IgnoreCase);
		}
		
		public static bool IsGridBlockSizeName (string name)
		{
			return Regex.IsMatch(name, "num_groups_[x|y|z]", RegexOptions.IgnoreCase);
		}
	}
}

