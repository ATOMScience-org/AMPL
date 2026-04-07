#!/usr/bin/env bash
#
# Purpose:
#     regenerates uv.lock.<platform>
#
# Use when: 
#     dependencies changed, pyproject.toml changed, or lockfile mismatch happens. refresh the commited lockfile.
#
# Usage:
#     ./update_lock.sh <cpu|cuda|rocm|mchip>

set -euo pipefail

platform="${1:?usage: $0 <cpu|cuda|rocm|mchip>}"

lockfile="uv.lock.${platform}"
venv_dir=".venv-${platform}"

rm -f uv.lock
rm -rf "$venv_dir"

case "$platform" in
  cpu)
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cpu --group dev
    ;;
  cuda)
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cuda --group dev
    ;;
  rocm)
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/rocm5.6 torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra rocm --group dev
    ;;
  mchip)
    uv venv --python 3.10 "$venv_dir"
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra mchip --group dev
    ;;
  *)
    echo "Invalid platform: $platform"
    exit 1
    ;;
esac

cp uv.lock "$lockfile"
echo "Updated $lockfile"
