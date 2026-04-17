#!/usr/bin/env bash
# Purpose:
#    syncs the environment, does not save a platform lockfile
#
# Use when:
#    want to create/update a local env from existing lock/deps. build an env to run.
#
# Usage:
#    ./sync_uv_env.sh <cpu|cuda|rocm|mchip>

usage() {
  echo "Usage: $0 <cpu|cuda|rocm|mchip>"
}

platform="${1:-}"
[[ -n "$platform" ]] || { usage; exit 1; }

case "$platform" in
  cpu)
    extra="cpu"
    torch_index="https://download.pytorch.org/whl/cpu"
    bootstrap_torch=true
    ;;
  cuda)
    extra="cuda"
    torch_index="https://download.pytorch.org/whl/cu121"
    bootstrap_torch=true
    ;;
  rocm)
    extra="rocm"
    torch_index="https://download.pytorch.org/whl/rocm5.6"
    bootstrap_torch=true
    ;;
  mchip)
    extra="mchip"
    torch_index=""
    bootstrap_torch=false
    ;;
  *)
    echo "Invalid platform: $platform"
    usage
    exit 1
    ;;
esac

venv_dir=".venv-${platform}"
lockfile="uv.lock.${platform}"
temp_lock="uv.lock"

[[ -f "$lockfile" ]] || {
  echo "Missing lockfile: $lockfile"
  echo "Run ./update_uv_lock.sh $platform first"
  exit 1
}

cleanup() {
  rm -f "$temp_lock"
}
trap cleanup EXIT

cp "$lockfile" "$temp_lock"
rm -rf "$venv_dir"

uv venv --python 3.10 "$venv_dir"

if [[ "$bootstrap_torch" == true ]]; then
  uv pip install --python "$venv_dir/bin/python" \
    --index-url "$torch_index" \
    torch==2.1.2 torchdata==0.7.1
fi

UV_PROJECT_ENVIRONMENT="$venv_dir" \
  uv sync --python "$venv_dir/bin/python" --group dev --extra "$extra" --locked

uv pip install --python "$venv_dir/bin/python" -e . --no-deps

echo "Refreshed $venv_dir from $lockfile"
echo "Activate with: source $venv_dir/bin/activate"