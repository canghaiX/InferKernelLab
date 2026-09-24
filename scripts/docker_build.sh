#!/usr/bin/env bash
set -euo pipefail

docker build --build-arg BASE_IMAGE="${BASE_IMAGE:-nano-vllm:optimized}" \
  -t inferkernellab:cuda .
