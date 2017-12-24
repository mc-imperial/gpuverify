#!/bin/bash
set -ev

mkdir -p ${DOWNLOADS_DIR}
cd ${DOWNLOADS_DIR}
wget -c http://releases.llvm.org/${LLVM_FULL_VERSION}/${LLVM}.tar.xz
if [ ! -d "${LLVM}" ]; then
  tar xvfJ ${LLVM}.tar.xz
fi
