#!/bin/bash

set -ex

: "${PYTHON:=python3}"
PACKAGE="atomsci"
APP="ampl"

INSTALL="--user"
if [ "$1" = "system" ]; then
    INSTALL=""
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "DIR: $DIR"
cd "$DIR"

TOPDIR="$(readlink -f .)"
DIST_DIR="${TOPDIR}/dist"

mkdir -p "$DIST_DIR"

# Ensure the package is in the DIST_DIR
if [ ! -f "${DIST_DIR}/${PACKAGE}_${APP}"* ]; then
    echo "ERROR: ${PACKAGE}_${APP} package not found in ${DIST_DIR}"
    exit 1
fi

$PYTHON -m pip install --pre --upgrade --no-index --find-links="$DIST_DIR" --no-deps "${PACKAGE}_${APP}" $INSTALL --force-reinstall -I -v || exit 1
