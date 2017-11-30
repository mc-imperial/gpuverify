#!/bin/bash
set -ev

mkdir -p ${DOWNLOADS_DIR}
cd ${DOWNLOADS_DIR}
wget -c \
  https://github.com/Z3Prover/z3/releases/download/z3-${Z3_VERSION}/${Z3}.zip
if [ ! -d "${Z3}" ]; then
  unzip ${Z3}.zip
fi
cd ${BUILD_ROOT}
ln -s ${DOWNLOADS_DIR}/${Z3}/bin/z3 z3.exe
