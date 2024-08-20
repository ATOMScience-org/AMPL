#!/usr/bin/env bash

# set env
export ENABLE_LIVERMORE=1
BASEDIR=$(dirname $(realpath "$0"))

# run unit tests
cd $BASEDIR/unit
pytest -ra -v -m "skipif" 
