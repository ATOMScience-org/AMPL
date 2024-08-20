#!/usr/bin/env bash

# set env
export ENABLE_LIVERMORE=1

# run unit tests
cd unit
pytest -s -v -m "skipif" 
