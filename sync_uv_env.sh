#!/usr/bin/env bash
# Purpose:
#    syncs the environment, does not save a platform lockfile
#
# Use when:
#    want to create/update a local env from existing lock/deps. build an env to run.
#
# Usage:
#    ./sync_uv_env.sh <cpu|cuda|rocm|mchip>

set -euo pipefail

platform="${1:?usage: $0 <cpu|cuda|rocm|mchip>}"

lockfile="uv.lock.${platform}"
venv_dir=".venv-${platform}"

if [[ ! -f "$lockfile" ]]; then
  echo "Missing lockfile: $lockfile"
  echo "Run ./update_lock.sh $platform first"
  exit 1
fi

cp "$lockfile" uv.lock
rm -rf "$venv_dir"

case "$platform" in
  cpu)
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cpu --group dev --locked
    ;;
  cuda)
    # to force the right torch wheel into the env before syncing, especially for GPU variants
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cuda --group dev --locked
    ;;
  rocm)
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/rocm5.6 torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra rocm --group dev --locked
    ;;
  mchip)
    uv venv --python 3.10 "$venv_dir"
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra mchip --group dev --locked
    ;;
  *)
    echo "Invalid platform: $platform"
    exit 1
    ;;
esac

echo "Synced $venv_dir from $lockfile"
