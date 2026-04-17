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
venv_dir=".venv-${platform}"
temp_lock="uv.lock"

usage() {
  echo "Usage: $0 <cpu|cuda|rocm|mchip>"
}

case "$platform" in
  cpu)
    extra="cpu"
    torch_index="https://download.pytorch.org/whl/cpu"
    bootstrap_torch="yes"
    ;;
  cuda)
    extra="cuda"
    torch_index="https://download.pytorch.org/whl/cu121"
    bootstrap_torch="yes"
    ;;
  rocm)
    extra="rocm"
    torch_index="https://download.pytorch.org/whl/rocm5.6"
    bootstrap_torch="yes"
    ;;
  mchip)
    extra="mchip"
    torch_index=""
    bootstrap_torch="no"
    ;;
  "")
    usage
    exit 1
    ;;
  *)
    echo "Invalid platform: $platform"
    usage
    exit 1
    ;;
esac

lockfile="uv.lock.${platform}"

cleanup() {
  rm -f "$temp_lock"
}
trap cleanup EXIT

rm -rf "$venv_dir"
uv venv --python 3.10 "$venv_dir"

if [[ "$bootstrap_torch" == "yes" ]]; then
  uv pip install --python "$venv_dir/bin/python" \
    --index-url "$torch_index" \
    torch==2.1.2 torchdata==0.7.1
fi

UV_PROJECT_ENVIRONMENT="$venv_dir" \
  uv sync --python "$venv_dir/bin/python" --extra "$extra" --group dev

cp "$temp_lock" "$lockfile"
echo "Updated $lockfile"