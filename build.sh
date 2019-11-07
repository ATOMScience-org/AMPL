#!/bin/bash

APP=ampl

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

TOPDIR=`readlink -f .`

BUILD_DIR=${TOPDIR}.build/$APP
DIST_DIR=${TOPDIR}.dist

mkdir -p $DIST_DIR
mkdir -p $BUILD_DIR

python3 setup.py build -b $BUILD_DIR egg_info --egg-base $BUILD_DIR bdist_wheel --dist-dir $DIST_DIR  || exit 1
