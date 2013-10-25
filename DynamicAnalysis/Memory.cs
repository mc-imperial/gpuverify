//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.Text.RegularExpressions;
using System.Linq;
using Microsoft.Boogie;

namespace DynamicAnalysis
{
    class UnitialisedException : Exception
    {
        public UnitialisedException(string message)
			: base(message)
        { 
        }
    }

    class Memory
    {
        private static Random Random = new Random();
        private Dictionary<string, BitVector> scalars = new Dictionary<string, BitVector>();
        private Dictionary<string, Dictionary <SubscriptExpr, BitVector>> arrays = new Dictionary<string, Dictionary <SubscriptExpr, BitVector>>();
        private Dictionary<string, HashSet<BitVector>> raceArrayOffsets = new Dictionary<string, HashSet<BitVector>>();
       
        public Memory()
        {
        }

        public void Clear()
        {
            scalars.Clear();
            arrays.Clear();
        }

        public void ClearRaceArrayOffset(string name)
        {
            raceArrayOffsets[name].Clear();
        }

        public bool HadRaceArrayVariable(string name)
        {
            return raceArrayOffsets.ContainsKey(name);
        }

        public void AddRaceArrayVariable(string name)
        {
            raceArrayOffsets[name] = new HashSet<BitVector>();
        }

        public void AddRaceArrayOffset(string name, BitVector offset)
        {
            raceArrayOffsets[name].Add(offset);
        }

        public List<string> GetRaceArrayVariables()
        {
            return raceArrayOffsets.Keys.ToList();
        }

        public HashSet<BitVector> GetRaceArrayOffsets(string name)
        {
            return raceArrayOffsets[name];
        }

        public void AddGlobalArray(string name)
        {
            arrays[name] = new Dictionary<SubscriptExpr, BitVector>();
        }

        public void Store(string name, BitVector val)
        {
            scalars[name] = val;
        }

        public bool Contains(string name)
        {
            return scalars.ContainsKey(name);
        }
        
        public bool Contains(string name, SubscriptExpr subscript)
        {
            if (arrays.ContainsKey(name))
            {
                Dictionary <SubscriptExpr, BitVector> arrayLocations = arrays[name];
                foreach (KeyValuePair<SubscriptExpr, BitVector> item in arrayLocations)
                {
                    if (SubscriptExpr.Matches(item.Key, subscript))
                        return true;
                }
            }
            return false;
        }

        public void Store(string name, SubscriptExpr subscript, BitVector val)
        {
            if (!arrays.ContainsKey(name))
                arrays[name] = new Dictionary<SubscriptExpr, BitVector>();
            if (!SubscriptExpr.Matches(subscript, arrays[name].Keys.ToList()))
                arrays[name][subscript] = val;
        }

        public BitVector GetValue(string name)
        {
            if (scalars.ContainsKey(name))
                return scalars[name];
            throw new UnitialisedException(String.Format("Location '{0}' has not been initialised", name));
        }

        public BitVector GetValue(string name, SubscriptExpr subscript)
        {
            Print.ConditionalExitMessage(arrays.ContainsKey(name), String.Format("Unable to find array '{0}' in memory", name));
            Dictionary <SubscriptExpr, BitVector> arrayLocations = arrays[name];
            foreach (KeyValuePair<SubscriptExpr, BitVector> item in arrayLocations)
            {
                if (SubscriptExpr.Matches(item.Key, subscript))
                    return arrays[name][item.Key];
            }
            Print.WarningMessage(String.Format("Location '{0}' in array '{1}' has not been initialised", subscript.ToString(), name));
            // Assign a random value
            int lowestVal = (int)-Math.Pow(2, 32 - 1);
            int highestVal = (int)Math.Pow(2, 32 - 1) - 1;
            BitVector val = new BitVector(Random.Next(lowestVal, highestVal));
            arrays[name][subscript] = val;
            return val;
        }

        private string getEmptySpaces(int maxLength, int length)
        {
            int size = maxLength - length;
            StringBuilder sb = new StringBuilder(size);
            for (int i = 0; i < size; ++i)
                sb.Append(" ");
            return sb.ToString();
        }

        public void Dump()
        {
            int maxLength = 0;
            foreach (string name in scalars.Keys.ToList())
                maxLength = Math.Max(maxLength, name.Length);
			
            Console.WriteLine("===== Scalar memory contents =====");
            foreach (KeyValuePair<string, BitVector> item in scalars)
                Console.WriteLine(item.Key
                + getEmptySpaces(maxLength, item.Key.Length)
                + " = "
                + item.Value.ToString());
            Console.WriteLine("==================================");
			
            Console.WriteLine("===== Array memory contents ======");
            foreach (KeyValuePair<string, Dictionary <SubscriptExpr, BitVector>> item in arrays)
            {
                foreach (KeyValuePair<SubscriptExpr, BitVector> item2 in item.Value)
                    Console.WriteLine(item.Key +
                    "[" +
                    item2.Key.ToString() +
                    "] = " +
                    item2.Value.ToString());
            }
            Console.WriteLine("==================================");
			
            maxLength = 0;
            foreach (string name in raceArrayOffsets.Keys.ToList())
                maxLength = Math.Max(maxLength, name.Length);
            Console.WriteLine("=========== Race-checking sets ===========");
            foreach (KeyValuePair<string, HashSet<BitVector>> item in raceArrayOffsets)
            {
                Console.Write(item.Key + getEmptySpaces(maxLength, item.Key.Length));
                Console.Write(" = {");
                int i = 0;
                foreach (BitVector offset in item.Value)
                {
                    Console.Write(offset.ToString());
                    if (++i < item.Value.Count)
                        Console.Write(", ");
                }
                Console.WriteLine("}");
            }
            Console.WriteLine("==================================");
        }
    }

    class SubscriptExpr
    {
        private List<BitVector> indices = new List<BitVector>();

        public static bool Matches(SubscriptExpr expr1, SubscriptExpr expr2)
        {
            if (expr1.indices.Count != expr2.indices.Count)
                return false;
            foreach (var pair in expr1.indices.Zip(expr2.indices))
            {
                if (pair.Item1 != pair.Item2)
                    return false;
            }
            return true;
        }

        public static bool Matches(SubscriptExpr expr, List<SubscriptExpr> exprs)
        {
            foreach (SubscriptExpr expr2 in exprs)
            {
                if (Matches(expr, expr2))
                    return true;
            }
            return false;
        }

        public SubscriptExpr()
        {
        }

        public void AddIndex(BitVector index)
        {
            indices.Add(index);
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            int i = 1;
            foreach (BitVector idx in indices)
            {
                builder.Append(idx);
                if (++i < indices.Count)
                    builder.Append(", ");
            }
            return builder.ToString();
        }
    }
}

