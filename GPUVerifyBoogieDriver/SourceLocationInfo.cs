//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Microsoft.Boogie;

namespace GPUVerify {

  class SourceLocationInfo {

    private string file;
    private string directory;
    private int line;
    private int column;

    public SourceLocationInfo(QKeyValue attributes, IToken fallBackToken) {
      try {
        file = QKeyValue.FindStringAttribute(attributes, "fname");
        directory = QKeyValue.FindStringAttribute(attributes, "dir");
        line = QKeyValue.FindIntAttribute(attributes, "line", -1);
        column = QKeyValue.FindIntAttribute(attributes, "col", -1);

        if(file == null || directory == null || line == -1 || column == -1) {
          throw new Exception();
        }

      } catch(Exception) {
        file = fallBackToken.filename;
        directory = "";
        line = fallBackToken.line;
        column = fallBackToken.col;
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
