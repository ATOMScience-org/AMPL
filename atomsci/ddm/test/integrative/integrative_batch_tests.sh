#!/usr/bin/env bash

# walk the directory tree and run pytest
for f in *; do
    if [ -d "$f" ] ; then
         # look for the directories that contain test*.py. if found, cd into and run pytest
         file=($f/test_*.py)
         if [[ -f "$file" ]]; then
            cd ${f}
            echo "Testing $f"
            pytest
            cd ..
        fi
    fi
done
