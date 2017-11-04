#!/bin/bash
set -ev

cd ${TRAVIS_BUILD_DIR}
ln -s `which clang-${LLVM_VERSION}` clang
ln -s `which opt-${LLVM_VERSION}` opt
ln -s `which llvm-nm-${LLVM_VERSION}` llvm-nm
