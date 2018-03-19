#!/bin/bash
set -ev

if [ -z ${DOWNLOADS_DIR+x} ]; then
  echo "DOWNLOADS_DIR not set"
  exit 1
fi

if [ -z ${LLVM_VERSION+x} ]; then
  echo "LLVM_VERSION not set"
  exit 1
fi

if [ -z ${LLVM+x} ]; then
  echo "LLVM not set"
else
  export LLVM_CONFIG="${DOWNLOADS_DIR}/${LLVM}/bin/llvm-config"
  ${GPUVERIFY_DIR}/.travis/download_llvm.sh
  cp ${GPUVERIFY_DIR}/.travis/gvfindtools.downloaded_llvm.py \
    ${GPUVERIFY_DIR}/gvfindtools.py
  cd ${BUILD_ROOT}
  git clone --depth=${CLONE_DEPTH} --branch="release_${LLVM_VERSION//.}" \
    https://llvm.org/git/llvm.git
  cd llvm/tools
  git clone --depth=${CLONE_DEPTH} --branch="release_${LLVM_VERSION//.}" \
    https://llvm.org/git/clang.git
fi

if [ "${GPUVERIFY_DIR}" == "${BUILD_ROOT}" ]; then
  ${GPUVERIFY_DIR}/.travis/build_gpuverify.sh
  ${GPUVERIFY_DIR}/.travis/build_bugle.sh
else
  echo "Unexpected BUILD_ROOT: ${BUILD_ROOT}"
  exit 1
fi

${GPUVERIFY_DIR}/.travis/clone_build_and_install_libclc.sh

if [ -z ${Z3+x} ]; then
  echo "Z3 not set"
  exit 1
else
  ${GPUVERIFY_DIR}/.travis/download_and_symlink_z3.sh
  cd ${BUILD_ROOT}
  git clone --depth=${CLONE_DEPTH} --branch="z3-${Z3_VERSION}" \
    https://github.com/Z3Prover/z3.git
fi

if [ -z ${CVC4+x} ]; then
  echo "CVC4 not set"
  exit 1
else
  ${GPUVERIFY_DIR}/.travis/download_and_symlink_cvc4.sh
  cd ${BUILD_ROOT}
  git clone --depth=${CLONE_DEPTH} --branch="${CVC4_VERSION}" \
    https://github.com/CVC4/CVC4.git
fi

cd ${BUILD_ROOT}
mkdir ${DEPLOY_DIR}
$PYTHON ${GPUVERIFY_DIR}/deploy.py ${DEPLOY_DIR}
zip -r GPUVerifyLinux64.zip ${DEPLOY_DIR}
zip -r libclc.zip libclc-install

