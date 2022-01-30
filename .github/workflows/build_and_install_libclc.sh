#!/bin/bash
set -ev

cd ${LLVM_PROJECT_DIR}/libclc
$PYTHON ./configure.py -g ninja --with-llvm-config=${LLVM_CONFIG} \
  --with-cxx-compiler=${CXX} --prefix=${LIBCLC_INSTALL_DIR} \
  nvptx-- nvptx64--
ninja install
