#!/bin/bash
set -ev

wget -c https://github.com/cvc5/cvc5/releases/download/${CVC4_VERSION}/${CVC4}
chmod u+x ${CVC4}
ln -s ${CVC4} cvc4.exe
