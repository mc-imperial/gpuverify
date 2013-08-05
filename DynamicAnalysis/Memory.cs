using System;
using System.Collections.Generic;
using System.Collections.Specialized;

namespace DynamicAnalysis
{		
	class Memory
	{
		private Dictionary<string, BitVector32> bvLocations = new Dictionary<string, BitVector32>();
		
		public Memory ()
		{
		}
		
		public void Clear ()
		{
			bvLocations.Clear();
		}
		
		public void Store (string name, BitVector32 val)
		{
			bvLocations[name] = val;
		}
		
		public BitVector32 GetValue (string name)
		{
			return bvLocations[name];
		}
		
		public void Dump ()
		{
			Console.WriteLine("===== Memory contents =====");
			foreach (KeyValuePair<string, BitVector32> item in bvLocations)
			{
				Console.WriteLine(item.Key + " " + Convert.ToString(item.Value.Data));
			}
		}
	}
}

