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
        public static BitVector False = new BitVector(0, 1);
        public static BitVector True = new BitVector(1, 1);
        public static BitVector Max32Int = new BitVector((int)Math.Pow(2, 32) - 1);
        public string Bits;

        public BitVector(int val, int width = 32)
        {
            Bits = Convert.ToString(val, 2);
            if (width > 1)
                SignExtend(width);
            Print.ConditionalExitMessage(CheckWidthIsPowerOfTwo(), "BV width is not a power of 2");
        }
        
        private BitVector (string bits)
        {
            Bits = bits;
            Print.ConditionalExitMessage(CheckWidthIsPowerOfTwo(), "BV width is not a power of 2");
        }

        public BitVector (BvConst bv)
        {
            // Create bit-string representation
            string str = bv.ToReadableString();
            string bareStr = str.Substring(0, str.IndexOf("bv"));
            if (bareStr.StartsWith("0x"))
            {
                bareStr = bareStr.Replace("0x", "").Replace(".", "");
                for (int i = 0; i < bareStr.Length; ++i)
                {
                    Bits += HexToBinary(bareStr[i]);
                }
            }
            else
            {
                int val = Convert.ToInt32(bareStr);
                Bits = Convert.ToString(val, 2);
            }
            SignExtend(bv.Bits);
            Print.ConditionalExitMessage(CheckWidthIsPowerOfTwo(), "BV width is not a power of 2");
        }

        private void SignExtend (int width)
        {
            if (Bits.Length < width)
            {
                char sign = Bits[0];
                Bits = Bits.PadLeft(width, '0');
            }
        }
        
        private bool CheckWidthIsPowerOfTwo ()
        {
            int width = Bits.Length;
            return (width != 0) && ((width & (width - 1)) == 0);
        }

        private string HexToBinary(char hex)
        {
            switch (hex)
            {
                case '0':
                    return "0000";
                case '1':
                    return "0001";
                case '2':
                    return "0010";
                case '3':
                    return "0011";
                case '4':
                    return "0100";
                case '5':
                    return "0101";
                case '6':
                    return "0110";
                case '7':
                    return "0111";
                case '8':
                    return "1000";
                case '9':
                    return "1001";
                case 'a':
                case 'A':
                    return "1010";
                case 'b':
                case 'B':
                    return "1011";
                case 'c':
                case 'C':
                    return "1100";
                case 'd':
                case 'D':
                    return "1101";
                case 'e':
                case 'E':
                    return "1110";
                case 'f':
                case 'F':
                    return "1111";
                default:
                    Print.ExitMessage("Unhandled hex character " + hex);
                    return "";  
            }
        }
        
        public int ConvertToInt32 ()
        {
           try
            {
                int data = Convert.ToInt32(Bits, 2);
                return data;
            }
            catch (OverflowException)
            {
                throw;
            }
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            BitVector item = obj as BitVector;
            if ((object)item == null)
                return false;
            return this.Bits.Equals(item.Bits);
        }

        public override int GetHashCode()
        {
            return Bits.GetHashCode();
        }

        public override string ToString()
        {
            return Bits;
        }

        public static BitVector operator+(BitVector a, BitVector b)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                val = checked(aData + bData);
            }
            catch (OverflowException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator-(BitVector a, BitVector b)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                val = checked(aData - bData);
            }
            catch (OverflowException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator*(BitVector a, BitVector b)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                val = checked(aData * bData);
            }
            catch (OverflowException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator/(BitVector a, BitVector b)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                val = checked(aData / bData);
            }
            catch (DivideByZeroException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator%(BitVector a, BitVector b)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                val = checked(aData % bData);
            }
            catch (DivideByZeroException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator&(BitVector a, BitVector b)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                val = checked(aData & bData);
            }
            catch (OverflowException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator|(BitVector a, BitVector b)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                val = checked(aData | bData);
            }
            catch (OverflowException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator>>(BitVector a, int shift)
        {
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                val = checked(aData >> shift);
            }
            catch (OverflowException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static BitVector operator<<(BitVector a, int shift)
        {            
            int val;
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                val = checked(aData << shift);
            }
            catch (OverflowException)
            {
                throw;
            }
            return new BitVector(val);
        }

        public static bool operator==(BitVector a, BitVector b)
        {
            return a.Bits.Equals(b.Bits);
        }

        public static bool operator!=(BitVector a, BitVector b)
        {
            return !a.Bits.Equals(b.Bits);
        }

        public static bool operator<(BitVector a, BitVector b)
        {
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                return aData < bData;
            }
            catch (OverflowException)
            {
                throw;
            }
        }

        public static bool operator<=(BitVector a, BitVector b)
        {
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                return aData <= bData;
            }
            catch (OverflowException)
            {
                throw;
            }
        }

        public static bool operator>(BitVector a, BitVector b)
        {
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                return aData > bData;
            }
            catch (OverflowException)
            {
                throw;
            }
        }

        public static bool operator>=(BitVector a, BitVector b)
        {
            try
            {
                int aData = Convert.ToInt32(a.Bits, 2);
                int bData = Convert.ToInt32(b.Bits, 2);
                return aData >= bData;
            }
            catch (OverflowException)
            {
                throw;
            }
        }
        
        public static BitVector operator^(BitVector a, BitVector b)
        {
            Print.ConditionalExitMessage(a.Bits.Length == b.Bits.Length, "The XOR operator requires bit vectors of equal width"); 
            char[] bits = new char[a.Bits.Length];
            for (int i = 0; i < a.Bits.Length; ++i)
            {
                if (a.Bits[i] == '1' && b.Bits[i] == '0')
                    bits[i] = '1';
                else if (a.Bits[i] == '1' && b.Bits[i] == '0')
                    bits[i] = '1';
                else
                    bits[i] = '0';
            }
            return new BitVector(new string(bits));
        }
        
        public static BitVector Slice (BitVector a, int high, int low)
        {
            Print.ConditionalExitMessage(high > low, "Slicing " + a.ToString() + 
            " is not defined because the slice [" + high.ToString() + ":" + low.ToString() + "] is not defined"); 
            int startIndex = a.Bits.Length - high;
            int length = high - low;
            string bits = a.Bits.Substring(startIndex, length);
            return new BitVector(bits);
        }
        
        public static BitVector Concatenate (BitVector a, BitVector b)
        {
            string bits = a.Bits + b.Bits;
            return new BitVector(bits);
        }
        
        public static BitVector LogicalShiftRight(BitVector a, int shift)
        {
            string bits = a.Bits.Substring(0, a.Bits.Length - shift);
            bits = bits.PadLeft(a.Bits.Length, '0');
            return new BitVector(bits);
        }
        
        public static BitVector ZeroExtend (BitVector a, int width)
        {
            string bits = a.Bits;
            bits = bits.PadLeft(width, '0');
            return new BitVector(bits);
        }
    }
}

