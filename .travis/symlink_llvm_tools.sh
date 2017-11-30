#!/bin/bash
set -ev

cd ${BUILD_ROOT}
ln -s `which clang-${LLVM_VERSION}` clang
ln -s `which opt-${LLVM_VERSION}` opt
ln -s `which llvm-nm-${LLVM_VERSION}` llvm-nm
