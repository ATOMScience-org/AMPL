#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

# --- begin: must be sourced guard ---
(return 0 2>/dev/null) || {
    echo "b This script must be sourced, not executed."
    echo "   Use:  source ${BASH_SOURCE[0]}  or  . ${BASH_SOURCE[0]}"
    exit 1
}
# --- end: must be sourced guard ---

PY_VER=3.10
PY_VER_STR="${PY_VER//./}"

# load system python module
echo module load python/$PY_VER
module load python/$PY_VER || return 1

# activate the venv
VENV=$(realpath venvs/ampl-py${PY_VER_STR}-venv/bin/activate) || return 1
echo "Activating ampl-py${PY_VER_STR}-venv @ $VENV"
source venvs/ampl-py${PY_VER_STR}-venv/bin/activate || return 1



