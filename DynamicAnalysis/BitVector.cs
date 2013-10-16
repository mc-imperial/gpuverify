//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System;
using System.Collections;
using Microsoft.Boogie;

namespace DynamicAnalysis
{
	public class BitVector
	{
		public static BitVector Zero = new BitVector(0);
		public static BitVector False = Zero;
		public static BitVector True = new BitVector(1);
		public static BitVector Max32Int = new BitVector((int) Math.Pow(2, 32) - 1);
		public string Bits;
		public int Data;
		public int Width;
		
		public BitVector (BvConst bv)
		{
			Print.ConditionalExitMessage(bv.Bits <= 32, "Unhandled bit vector width");
			string str = bv.ToString();
			string subStr = str.Substring(0, str.IndexOf('b'));
			long val;
			try
			{
				val = Convert.ToInt32(subStr);
			}
			catch (OverflowException)
			{
				val = Convert.ToInt64(subStr);
			}
			
			Bits = Convert.ToString(val, 2);
			// Sign extend
			if (Bits.Length < bv.Bits)
				Bits = Bits.PadLeft(bv.Bits, '0');
			Data  = Convert.ToInt32(Bits, 2);
			Width = bv.Bits;
		}
		
		public BitVector (int val)
		{
			Bits = Convert.ToString(val, 2);
			// Sign extend
			if (Bits.Length < 32)
				Bits = Bits.PadLeft(32, '0');
			Data = val;
			Width = 32;
		}
		
		public override bool Equals (object obj)
		{
			if (obj == null)
				return false;
			BitVector item = obj as BitVector;
			if ((object) item == null)
				return false;
			return this.Data.Equals(item.Data);
		}
		
		public override int GetHashCode()
		{
			return Data.GetHashCode();
		}
				
		public override string ToString ()
		{
			return Data.ToString();
		}
		
		public static BitVector operator+ (BitVector a, BitVector b)
		{
			int val;
			try
			{
				val = checked(a.Data + b.Data);
			}
			catch (OverflowException)
			{
				throw;
			}
			return new BitVector(val);
		}
		
		public static BitVector operator- (BitVector a, BitVector b)
		{
			int val;
			try
			{
				val = checked(a.Data - b.Data);
			}
			catch (OverflowException)
			{
				throw;
			}
			return new BitVector(val);
		}
		
		public static BitVector operator* (BitVector a, BitVector b)
		{
			int val;
			try
			{
				val = checked(a.Data * b.Data);
			}
			catch (OverflowException)
			{
				throw;
			}
			return new BitVector(val);
		}
		
		public static BitVector operator/ (BitVector a, BitVector b)
		{
			int val;
			try
			{
				val = checked(a.Data / b.Data);
			}
			catch (DivideByZeroException)
			{
				throw;
			}
			return new BitVector(val);
		}
		
		public static BitVector operator% (BitVector a, BitVector b)
		{
			int val;
			try
			{
				val = checked(a.Data % b.Data);
			}
			catch (DivideByZeroException)
			{
				throw;
			}
			return new BitVector(val);
		}
		
		public static BitVector operator& (BitVector a, BitVector b)
		{
			return new BitVector(a.Data & b.Data);
		}
		
		public static BitVector operator| (BitVector a, BitVector b)
		{
			return new BitVector(a.Data | b.Data);
		}
		
		public static BitVector operator>> (BitVector a, int val)
		{
			return new BitVector(a.Data >> val);
		}
		
		public static BitVector operator<< (BitVector a, int val)
		{
			return new BitVector(a.Data << val);
		}
		
		public static bool operator== (BitVector a, BitVector b)
		{
			return a.Data == b.Data;
		}
		
		public static bool operator!= (BitVector a, BitVector b)
		{
			return a.Data != b.Data;
		}
		
		public static bool operator< (BitVector a, BitVector b)
		{
			return a.Data < b.Data;
		}
		
		public static bool operator<= (BitVector a, BitVector b)
		{
			return a.Data <= b.Data;
		}
		
		public static bool operator> (BitVector a, BitVector b)
		{
			return a.Data > b.Data;
		}
		
		public static bool operator>= (BitVector a, BitVector b)
		{
			return a.Data >= b.Data;
		}
	}
}

