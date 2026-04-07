#!/usr/bin/env bash
set -euo pipefail

platform="${1:?usage: $0 <cpu|cuda|rocm|mchip>}"

rm -f uv.lock

case "$platform" in
  cpu)
    venv_dir=".venv-cpu"
    rm -rf "$venv_dir"
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cpu --group dev
    ;;
  cuda)
    venv_dir=".venv-cuda"
    rm -rf "$venv_dir"
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra cuda --group dev
    ;;
  rocm)
    venv_dir=".venv-rocm"
    rm -rf "$venv_dir"
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --index-url https://download.pytorch.org/whl/rocm5.6 torch==2.1.2
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra rocm --group dev
    ;;
  mchip)
    venv_dir=".venv-mchip"
    rm -rf "$venv_dir"
    uv venv --python 3.10 "$venv_dir"
    VIRTUAL_ENV="$venv_dir" uv sync --python "$venv_dir/bin/python" --extra mchip --group dev
    ;;
  *)
    echo "Invalid platform: $platform"
    exit 1
    ;;
esac

cp uv.lock "uv.lock.${platform}"
