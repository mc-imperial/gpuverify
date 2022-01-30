#!/bin/bash
set -ev

wget -c \
  https://github.com/Z3Prover/z3/releases/download/z3-${Z3_VERSION}/${Z3}.zip
unzip ${Z3}.zip
ln -s ${Z3}/bin/z3 z3.exe
