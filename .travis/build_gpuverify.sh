#!/bin/bash
set -ev

cd ${GPUVERIFY_DIR}
nuget restore GPUVerify.sln
msbuild /m /p:Configuration=Release GPUVerify.sln
cp gvfindtools.templates/gvfindtools.travis.py gvfindtools.py
