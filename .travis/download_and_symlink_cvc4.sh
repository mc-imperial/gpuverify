#!/bin/bash
set -ev

mkdir -p ${DOWNLOADS_DIR}
cd ${DOWNLOADS_DIR}
wget -c http://cvc4.cs.stanford.edu/downloads/builds/x86_64-linux-opt/${CVC4}
chmod u+x ${CVC4}
cd ${BUILD_ROOT}
ln -s ${DOWNLOADS_DIR}/${CVC4} cvc4.exe
