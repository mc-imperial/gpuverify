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
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify {

  public class SourceLocationInfo {

    public class SourceLocationInfoComparison : IComparer<SourceLocationInfo> {
      public int Compare(SourceLocationInfo s1, SourceLocationInfo s2) {

        int directories = s1.GetDirectory().CompareTo(s2.GetDirectory());
        if(directories != 0) {
          return directories;
        }

        int files = s1.GetFile().CompareTo(s2.GetFile());
        if(files != 0) {
          return files;
        }

        if(s1.GetLine() < s2.GetLine()) {
          return -1;
        }

        if(s1.GetLine() > s2.GetLine()) {
          return 1;
        }

        if(s1.GetColumn() < s2.GetColumn()) {
          return -1;
        }

        if(s1.GetColumn() > s2.GetColumn()) {
          return 1;
        }

        return 0;
      }
    }

    private int line;
    private int column;
    private string file;
    private string directory;

    public SourceLocationInfo(QKeyValue attributes, string sourceFileName, IToken fallBackToken) {
      try {
        var sourceLocFileName = 
          Path.GetFileNameWithoutExtension(sourceFileName) + ".loc";
        using (StreamReader sr = new StreamReader(sourceLocFileName)) {
          int number = QKeyValue.FindIntAttribute(attributes, "sourceloc_num", -1);
          if(number == -1) {
            throw new Exception();
          }
          var info = sr.ReadLine().Split(new char[] { '\x1D' })[number];
          var chain = info.Split(new char[] { '\x1E' });
          Debug.Assert(chain[chain.Count() - 1] == "");
          var last = chain[chain.Count() - 2];
          var sourceInfo = last.Split(new char[] { '\x1F' });
          this.line = Convert.ToInt32(sourceInfo[0]);
          this.column = Convert.ToInt32(sourceInfo[1]);
          this.file = sourceInfo[2];
          this.directory = sourceInfo[3];
        }
      } catch (Exception) {
        // Don't warn, just fall back to Boogie token
        this.file = fallBackToken.filename;
        this.directory = "";
        this.line = fallBackToken.line;
        this.column = fallBackToken.col;
      }
    }

    public string GetFile() {
      return file;
    }

    public string GetDirectory() {
      return directory;
    }

    public int GetLine() {
      return line;
    }

    public int GetColumn() {
      return column;
    }

    public override string ToString() {
      return GetFile() + ":" + GetLine() + ":" + GetColumn() + ":";
    }

    public string FetchCodeLine() {
      if(File.Exists(GetFile())) {
        return FetchCodeLine(GetFile(), GetLine());
      }
      return FetchCodeLine(GetDirectory() + "\\" + Path.GetFileName(GetFile()), GetLine());
    }

    public static string FetchCodeLine(string path, int lineNo) {
      try {
        TextReader tr = new StreamReader(path);
        string line = null;
        for (int currLineNo = 1; ((line = tr.ReadLine()) != null); currLineNo++) {
          if (currLineNo == lineNo) {
            return line;
          }
        }
        throw new Exception();
      }
      catch (Exception) {
        return "<unknown line of code>";
      }
    }

  }


}
