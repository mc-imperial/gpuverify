using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace DynamicAnalysis
{
	public enum DIMENSION {X, Y, Z};
	
	public class GPU
	{
		public Dictionary<DIMENSION, int> gridDim;
		public Dictionary<DIMENSION, int> blockDim;
		
		public GPU ()
		{
			Clear();
		}
		
		public void Clear ()
		{
			gridDim = new Dictionary<DIMENSION, int>();
			blockDim = new Dictionary<DIMENSION, int>();
			gridDim[DIMENSION.X] = 1;
			gridDim[DIMENSION.Y] = 1;
			gridDim[DIMENSION.Z] = 1;
			blockDim[DIMENSION.X] = 1;
			blockDim[DIMENSION.Y] = 1;
			blockDim[DIMENSION.Z] = 1;
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

