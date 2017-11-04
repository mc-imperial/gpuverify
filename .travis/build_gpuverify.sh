#!/bin/bash
set -ev

cd ${GPUVERIFY_DIR}
xbuild /p:Configuration=Release GPUVerify.sln
cp gvfindtools.templates/gvfindtools.travis.py gvfindtools.py
