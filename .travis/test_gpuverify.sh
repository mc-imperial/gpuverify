#!/bin/bash
set -ev

cd ${GPUVERIFY_DIR}
$PYTHON ./gvtester.py --write-pickle run.pickle testsuite
$PYTHON ./gvtester.py --compare-pickle testsuite/baseline.pickle run.pickle
