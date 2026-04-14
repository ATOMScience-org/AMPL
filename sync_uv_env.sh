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

platform="${1:-}"

case "$platform" in
  cpu|cuda|rocm|mchip)
    ;;
  "")
    echo "Usage: $0 <cpu|cuda|rocm|mchip>"
    exit 1
    ;;
  *)
    echo "Invalid platform: $platform"
    echo "Supported platforms: cpu cuda rocm mchip"
    echo "Usage: $0 <cpu|cuda|rocm|mchip>"
    exit 1
    ;;
esac

lockfile="uv.lock.${platform}"
venv_dir=".venv-${platform}"

if [[ ! -f "$lockfile" ]]; then
  echo "Missing lockfile: $lockfile"
  echo "Run ./update_uv_lock.sh $platform first"
  exit 1
fi

cp "$lockfile" uv.lock
rm -rf "$venv_dir"

uv venv --python 3.10 "$venv_dir"

case "$platform" in
  cpu)
    uv pip install --python "$venv_dir/bin/python" \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.1.2 torchdata==0.7.1
    UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cpu --group dev --locked
    ;;
  cuda)
    uv pip install --python "$venv_dir/bin/python" \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.1.2 torchdata==0.7.1
    UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cuda --group dev --locked
    ;;
  rocm)
    uv pip install --python "$venv_dir/bin/python" \
      --index-url https://download.pytorch.org/whl/rocm5.6 \
      torch==2.1.2 torchdata==0.7.1
    UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra rocm --group dev --locked
    ;;
  mchip)
    UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra mchip --group dev --locked
    ;;
  *)
    echo "Invalid platform: $platform"
    exit 1
    ;;
esac

# Install AMPL in editable mode without re-resolving dependencies.
uv pip install --python "$venv_dir/bin/python" -e . --no-deps

echo "Synced $venv_dir from $lockfile and installed AMPL editable (--no-deps)"
