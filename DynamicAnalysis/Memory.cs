using System;
using System.Collections.Generic;

namespace DynamicAnalysis
{	
	class Value<T>
	{	
		protected T val;
		
		public T getVal ()
		{
			return val;
		}
	}
	
	class IntVal : Value<int>
	{
		public IntVal (int val)
		{
			this.val = val;
		}
	}
	
	class Memory
	{
		private Dictionary<string, IntVal> integerLocations = new Dictionary<string, IntVal>();
		
		public Memory ()
		{
		}
		
		public void store (string name, int val)
		{
			integerLocations[name] = new IntVal(val);
		}
		
		public IntVal getValue (string name)
		{
			return integerLocations[name];
		}
		
		public void dump ()
		{
			Console.WriteLine("===== Memory contents =====");
			foreach (KeyValuePair<string,IntVal> item in integerLocations)
			{
				Console.WriteLine(item.Key + " " + Convert.ToString(item.Value.getVal()));
			}
		}
	}
}

