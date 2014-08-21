#!/bin/bash

KERNEL_NAME="kernel"

if [ $# -ne 1 ]; then
    echo "Usage $0 : <path>"
    echo "<path> - directory to recursively search"
    echo ""
    echo "This looks for GPUVerify temporary files and deletes them"
    exit
fi

DIR="$1"

if [ ! -d "${DIR}" ]; then
    echo "\"${DIR}\" is not a directory"
    exit 1
fi

set -x
find "${DIR}" \(  \
                 -iname "${KERNEL_NAME}.bpl" -o \
                 -iname "${KERNEL_NAME}.gbpl" -o \
                 -iname "${KERNEL_NAME}.cbpl" -o \
                 -iname "${KERNEL_NAME}.bc" -o \
                 -iname "${KERNEL_NAME}.opt.bc" -o \
                 -iname "${KERNEL_NAME}.loc" \
              \) \
              -exec rm --verbose  '{}' ';'
set +x
