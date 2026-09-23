#!/usr/bin/env bash
set -euo pipefail

docker build --build-arg BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}" \
  -t inferkernellab:cuda .
