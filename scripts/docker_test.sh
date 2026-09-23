#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  pytest -q
