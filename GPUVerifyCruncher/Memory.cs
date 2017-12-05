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
    using System.Linq;
    using System.Text;
    using Microsoft.Boogie;

    public enum MemorySpace
    {
        GLOBAL, GROUP_SHARED
    }

    public class Memory
    {
        private static Random random = new Random();
        private Dictionary<string, BitVector> scalars = new Dictionary<string, BitVector>();
        private Dictionary<string, Dictionary<SubscriptExpr, BitVector>> arrays = new Dictionary<string, Dictionary<SubscriptExpr, BitVector>>();
        private Dictionary<string, HashSet<BitVector>> raceArrayOffsets = new Dictionary<string, HashSet<BitVector>>();
        private Dictionary<string, MemorySpace> arrayLocations = new Dictionary<string, MemorySpace>();

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

        public bool HasRaceArrayVariable(string name)
        {
            return raceArrayOffsets.ContainsKey(name);
        }

        public void AddRaceArrayOffsetVariables(string name)
        {
            raceArrayOffsets["_WRITE_OFFSET_" + name] = new HashSet<BitVector>();
            raceArrayOffsets["_READ_OFFSET_" + name] = new HashSet<BitVector>();
            raceArrayOffsets["_ATOMIC_OFFSET_" + name] = new HashSet<BitVector>();
        }

        public void SetMemorySpace(string name, MemorySpace space)
        {
            arrayLocations[name] = space;
        }

        public bool IsInGlobalMemory(string name)
        {
            return arrayLocations.ContainsKey(name) && arrayLocations[name] == MemorySpace.GLOBAL;
        }

        public bool IsInGroupSharedMemory(string name)
        {
            return arrayLocations.ContainsKey(name) && arrayLocations[name] == MemorySpace.GROUP_SHARED;
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
                Dictionary<SubscriptExpr, BitVector> arrayLocations = arrays[name];
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
            SubscriptExpr matchingSubscript = SubscriptExpr.Matches(subscript, arrays[name].Keys.ToList());
            if (matchingSubscript != null)
                arrays[name].Remove(matchingSubscript);
            arrays[name][subscript] = val;
        }

        public BitVector GetValue(string name)
        {
            if (scalars.ContainsKey(name))
                return scalars[name];
            throw new UnitialisedException(string.Format("Location '{0}' has not been initialised", name));
        }

        public BitVector GetValue(string name, SubscriptExpr subscript)
        {
            Print.ConditionalExitMessage(arrays.ContainsKey(name), string.Format("Unable to find array '{0}' in memory", name));

            Dictionary<SubscriptExpr, BitVector> arrayLocations = arrays[name];
            foreach (KeyValuePair<SubscriptExpr, BitVector> item in arrayLocations)
            {
                if (SubscriptExpr.Matches(item.Key, subscript))
                    return arrays[name][item.Key];
            }

            Print.WarningMessage(string.Format("Location '{0}' in array '{1}' has not been initialised", subscript.ToString(), name));

            // Assign a random value
            BitVector val = new BitVector(random.Next(int.MinValue, int.MaxValue));
            arrays[name][subscript] = val;
            return val;
        }

        private string GetEmptySpaces(int maxLength, int length)
        {
            int size = maxLength - length;
            StringBuilder sb = new StringBuilder(size);
            for (int i = 0; i < size; ++i)
                sb.Append(" ");
            return sb.ToString();
        }

        public void Dump()
        {
            int maxLength = scalars.Keys.Aggregate(0, (curMax, name) => Math.Max(curMax, name.Length));

            Console.WriteLine("===== Scalar memory contents =====");

            foreach (KeyValuePair<string, BitVector> item in scalars)
            {
                Console.WriteLine(
                    item.Key + GetEmptySpaces(maxLength, item.Key.Length) +
                    " = " + item.Value.ToString());
            }

            Console.WriteLine("==================================");

            Console.WriteLine("===== Array memory contents ======");

            foreach (KeyValuePair<string, Dictionary<SubscriptExpr, BitVector>> item in arrays)
            {
                foreach (KeyValuePair<SubscriptExpr, BitVector> item2 in item.Value)
                {
                    Console.WriteLine(
                        item.Key + "[" + item2.Key.ToString() + "] = " +
                        item2.Value.ToString());
                }
            }

            Console.WriteLine("==================================");

            maxLength = raceArrayOffsets.Keys.Aggregate(0, (curMax, name) => Math.Max(curMax, name.Length));
            Console.WriteLine("=========== Race-checking sets ===========");

            foreach (KeyValuePair<string, HashSet<BitVector>> item in raceArrayOffsets)
            {
                Console.Write(item.Key + GetEmptySpaces(maxLength, item.Key.Length));
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

        public class SubscriptExpr
        {
            public List<BitVector> Indices { get; private set; } = new List<BitVector>();

            public SubscriptExpr()
            {
            }

            public static bool Matches(SubscriptExpr expr1, SubscriptExpr expr2)
            {
                if (expr1.Indices.Count != expr2.Indices.Count)
                    return false;

                foreach (var pair in expr1.Indices.Zip(expr2.Indices))
                {
                    if (!pair.Item1.Equals(pair.Item2))
                        return false;
                }

                return true;
            }

            public static SubscriptExpr Matches(SubscriptExpr expr, List<SubscriptExpr> exprs)
            {
                foreach (SubscriptExpr expr2 in exprs)
                {
                    if (Matches(expr, expr2))
                        return expr2;
                }

                return null;
            }

            public override string ToString()
            {
                StringBuilder builder = new StringBuilder();
                int i = 1;
                foreach (BitVector idx in Indices)
                {
                    builder.Append(idx);
                    if (++i < Indices.Count)
                        builder.Append(", ");
                }

                return builder.ToString();
            }
        }

        public class UnitialisedException : Exception
        {
            public UnitialisedException(string message)
                : base(message)
            {
            }
        }
    }
}
