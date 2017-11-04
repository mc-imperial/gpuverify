#!/bin/bash
set -ev

cd ${TRAVIS_BUILD_DIR}
git clone --depth=${CLONE_DEPTH} --branch=master https://llvm.org/git/libclc.git
cd libclc
$PYTHON ./configure.py -g ninja --with-llvm-config=${LLVM_CONFIG} \
  --with-cxx-compiler=c++ --prefix=${TRAVIS_BUILD_DIR}/libclc-install \
  nvptx-- nvptx64--
ninja install
