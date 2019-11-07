#!/bin/bash

PACKAGE=atomsci
APP=ampl

INSTALL=--user
if [ "$1" = "system" ]; then
    INSTALL=
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "DIR: $DIR"
cd $DIR

TOPDIR=`readlink -f .`

DIST_DIR=${TOPDIR}.dist

mkdir -p $DIST_DIR

python3 -m pip install --pre --upgrade --no-index --find-links=$DIST_DIR --no-deps ${PACKAGE}_${APP} $INSTALL --force-reinstall -I -v || exit 1
