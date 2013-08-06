using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.Linq;
using Microsoft.Boogie;

namespace DynamicAnalysis
{		
	class Memory
	{
		private Dictionary<string, BitVector32> scalars = new Dictionary<string, BitVector32>();
		private Dictionary<string, Dictionary <SubscriptExpr, BitVector32>> arrays = new Dictionary<string, Dictionary <SubscriptExpr, BitVector32>>();
		
		public Memory ()
		{
		}
		
		public void Clear ()
		{
			scalars.Clear();
			arrays.Clear();
		}
		
		public void Store (string name, BitVector32 val)
		{
			scalars[name] = val;
		}
		
		public void Store (string name, SubscriptExpr subscript, BitVector32 val)
		{
			if (!arrays.ContainsKey(name))
				arrays[name] = new Dictionary<SubscriptExpr, BitVector32>();
			if (!SubscriptExpr.Matches(subscript, arrays[name].Keys.ToList()))
				arrays[name][subscript] = val;
		}
		
		public BitVector32 GetValue (string name)
		{
			if (scalars.ContainsKey(name))
				return scalars[name];
			Print.ExitMessage(String.Format("Location '{0}' has not been initialised", name));
			return new BitVector32(0);
		}
		
		public BitVector32 GetValue (string name, SubscriptExpr subscript)
		{
			Dictionary <SubscriptExpr, BitVector32> arrayLocations = arrays[name];
			foreach (KeyValuePair<SubscriptExpr, BitVector32> item in arrayLocations)
			{
				if (SubscriptExpr.Matches(item.Key, subscript))
					return arrays[name][item.Key];
			}
			Print.ExitMessage(String.Format("Location '{0}' in array '{1}' has not been initialised", subscript.ToString(), name));
			return new BitVector32(0);
		}
		
		private string getEmptySpaces (int maxLength, int length)
		{
			int size = maxLength - length;
			StringBuilder sb = new StringBuilder(size);
			for (int i = 0; i < size; ++i)
				sb.Append(" ");
			return sb.ToString();
		}
		
		public void Dump ()
		{
			int maxLength = 0;
			foreach (string name in scalars.Keys.ToList())
				maxLength = Math.Max(maxLength, name.Length);
			
			Console.WriteLine("===== Scalar memory contents =====");
			foreach (KeyValuePair<string, BitVector32> item in scalars)
				Console.WriteLine(item.Key 
				                  + getEmptySpaces(maxLength, item.Key.Length) 
				                  + " = " 
				                  + Convert.ToString(item.Value.Data));
			
			Console.WriteLine("===== Array memory contents =====");
			foreach (KeyValuePair<string, Dictionary <SubscriptExpr, BitVector32>> item in arrays)
			{
				foreach (KeyValuePair<SubscriptExpr, BitVector32> item2 in item.Value)
					Console.WriteLine(item.Key + 
					                  "[" + 
					                  item2.Key.ToString() + 
					                  "] = " + 
					                  Convert.ToString(item2.Value.Data));
			}
		}
	}
	
	class SubscriptExpr
	{
		private List<BitVector32> indices = new List<BitVector32>();
		
		public static bool Matches (SubscriptExpr expr1, SubscriptExpr expr2)
		{
			if (expr1.indices.Count != expr2.indices.Count)
				return false;
			foreach (var pair in expr1.indices.Zip(expr2.indices))
			{
				if (pair.Item1.Data != pair.Item2.Data)
					return false;
			}
			return true;
		}
		
		public static bool Matches (SubscriptExpr expr, List<SubscriptExpr> exprs)
		{
			foreach (SubscriptExpr expr2 in exprs)
			{
				if (Matches(expr, expr2))
					return true;
			}
			return false;
		}
		
		public SubscriptExpr ()
		{
		}
		
		public void AddIndex (BitVector32 index)
		{
			indices.Add(index);
		}
		
		public override string ToString ()
		{
			StringBuilder builder = new StringBuilder();
			int i = 1;
			foreach (BitVector32 idx in indices)
			{
				builder.Append(idx.Data);
				if (++i < indices.Count)
					builder.Append(", ");
			}
			return builder.ToString();
		}
	}
}

