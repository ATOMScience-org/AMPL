#!/usr/bin/env bash
#
# Purpose:
#     regenerates uv.lock.<platform>
#     generates the righ lockfile for the specified platform
#
# Use when: 
#     dependencies changed, pyproject.toml changed, or lockfile mismatch happens. refresh the commited lockfile.
#
# Usage:
#     ./update_uv_lock.sh <cpu|cuda|rocm|mchip>

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

rm -f uv.lock
rm -rf "$venv_dir"

uv venv --python 3.10 "$venv_dir"

case "$platform" in
  cpu)
    uv pip install --python "$venv_dir/bin/python" \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.1.2 torchdata==0.7.1
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cpu --group dev
    ;;
  cuda)
    uv pip install --python "$venv_dir/bin/python" \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.1.2 torchdata==0.7.1
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cuda --group dev
    ;;
  rocm)
    uv pip install --python "$venv_dir/bin/python" \
      --index-url https://download.pytorch.org/whl/rocm5.6 \
      torch==2.1.2 torchdata==0.7.1
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra rocm --group dev
    ;;
  mchip)
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra mchip --group dev
    ;;
  *)
    echo "Invalid platform: $platform"
    exit 1
    ;;
esac

cp uv.lock "$lockfile"
echo "Updated $lockfile"
