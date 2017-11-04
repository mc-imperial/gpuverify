#!/bin/bash
set -ev

if [ "${GPUVERIFY_DIR}" == "${TRAVIS_BUILD_DIR}" ]; then
  ${GPUVERIFY_DIR}/.travis/build_gpuverify.sh
  ${GPUVERIFY_DIR}/.travis/build_bugle.sh
elif [ "${BUGLE_DIR}" == "${TRAVIS_BUILD_DIR}" ]; then
  ${GPUVERIFY_DIR}/.travis/build_bugle.sh
  ${GPUVERIFY_DIR}/.travis/build_gpuverify.sh
else
  echo "Unexpected TRAVIS_BUILD_DIR: ${TRAVIS_BUILD_DIR}"
  exit 1
fi

${GPUVERIFY_DIR}/.travis/clone_build_and_install_libclc.sh

if [ "${DEFAULT_SOLVER}" == "z3" ]; then
  ${GPUVERIFY_DIR}/.travis/download_and_symlink_z3.sh
fi

if [ "${DEFAULT_SOLVER}" == "cvc4" ]; then
  ${GPUVERIFY_DIR}/.travis/download_and_symlink_cvc4.sh
fi

${GPUVERIFY_DIR}/.travis/symlink_llvm_tools.sh
${GPUVERIFY_DIR}/.travis/test_gpuverify.sh
