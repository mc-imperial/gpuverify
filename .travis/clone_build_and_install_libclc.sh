#!/bin/bash
set -ev

cd ${BUILD_ROOT}
git clone --depth=${CLONE_DEPTH} --branch=master https://llvm.org/git/libclc.git
cd libclc
$PYTHON ./configure.py -g ninja --with-llvm-config=${LLVM_CONFIG} \
  --with-cxx-compiler=${CXX} --prefix=${BUILD_ROOT}/libclc-install \
  nvptx-- nvptx64--
ninja install
