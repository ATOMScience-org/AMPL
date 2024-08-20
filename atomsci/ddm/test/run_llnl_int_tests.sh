#!/usr/bin/env bash

# set env
export ENABLE_LIVERMORE=1
BASEDIR=$(dirname $(realpath "$0"))

# run integrative
cd $BASEDIR/integrative

TEST_FILES="$(grep -l 'skipif' ./*/test*.py)"

# split to array
array=(${TEST_FILES//$'\n'/ })

# iterate
for i in "${!array[@]}"
do
    file="${array[i]}"
    parentdir="$(dirname "$file")"
    cd $parentdir
    echo "Testing $file"
    pytest -ra -v -m "skipif"
    cd ..
done
