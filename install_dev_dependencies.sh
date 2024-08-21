#!/bin/bash
set -ex

if [[ "$1" == "dev" ]]; then
  echo "Development mode detected"
  if [ -f "./pip/dev_requirements.txt" ]; then
    pip install --no-cache-dir -r ./pip/dev_requirements.txt
  else
    echo "Error: ./pip/dev_requirements.txt not found."
    exit 1
  fi
else
  echo "$1"
  echo "ENV is not set to 'dev'. Skipping development requirements installation."
fi
