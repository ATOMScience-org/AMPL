#!/usr/bin/env bash

# set env
export ENABLE_LIVERMORE=1

# run integrative
cd integrative

# walk the directory tree and run pytest
for f in *; do
    if [ -d "$f" ] ; then
         # look for the directories that contain test*.py. if found, cd into and run pytest
         file=($f/test_*.py)
         if [[ -f "$file" ]]; then
            cd ${f}
            echo "Testing $f"
            pytest -s -v -m "skipif"
            cd ..
        fi
    fi
done
