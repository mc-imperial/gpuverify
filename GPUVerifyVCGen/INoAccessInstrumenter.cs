using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Boogie;

namespace GPUVerify
{
    interface INoAccessInstrumenter
    {
        void AddNoAccessInstrumentation();
    }
}
