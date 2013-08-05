using System;
using System.Collections.Generic;
using System.Collections.Specialized;

namespace DynamicAnalysis
{		
	class Memory
	{
		private Dictionary<string, BitVector32> scalarLocations = new Dictionary<string, BitVector32>();
		private Dictionary<string, Dictionary <SubscriptExpr, BitVector32>> arrayLocations = new Dictionary<string, Dictionary <SubscriptExpr, BitVector32>>();
		
		public Memory ()
		{
		}
		
		public void Clear ()
		{
			scalarLocations.Clear();
		}
		
		public void Store (string name, BitVector32 val)
		{
			scalarLocations[name] = val;
		}
		
		public void Store (string name, SubscriptExpr subscript, BitVector32 val)
		{
			if (!arrayLocations.ContainsKey(name))
				arrayLocations[name] = new Dictionary<SubscriptExpr, BitVector32>();
			arrayLocations[name][subscript] = val;
		}
		
		public BitVector32 GetValue (string name)
		{
			return scalarLocations[name];
		}
		
		public void Dump ()
		{
			Console.WriteLine("===== Memory contents =====");
			foreach (KeyValuePair<string, BitVector32> item in scalarLocations)
				Console.WriteLine(item.Key + " " + Convert.ToString(item.Value.Data));
			foreach (KeyValuePair<string, Dictionary <SubscriptExpr, BitVector32>> item in arrayLocations)
			{
				foreach (KeyValuePair<SubscriptExpr, BitVector32> item2 in item.Value)
					Console.WriteLine(item.Key + " " + Convert.ToString(item2.Value.Data));
			}
		}
	}
	
	class SubscriptExpr
	{
		private List<BitVector32> indices = new List<BitVector32>();
		
		public SubscriptExpr ()
		{
		}
		
		public void AddIndex (BitVector32 index)
		{
			indices.Add(index);
		}
	}
}

