#!/bin/bash
set -ev

cd ${BUGLE_DIR}
mkdir build && cd build
cmake -G Ninja -DLLVM_CONFIG_EXECUTABLE=`which ${LLVM_CONFIG}` \
  -DCMAKE_BUILD_TYPE=Release ..
ninja
