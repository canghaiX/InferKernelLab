ARG BASE_IMAGE=nano-vllm:optimized
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace
COPY pyproject.toml setup.py README.md ./
COPY src ./src
COPY tests ./tests
COPY docs ./docs
COPY scripts ./scripts

RUN python3 -m pip install --no-build-isolation -e '.[test,cuda]'

ENTRYPOINT []
CMD ["pytest", "-q"]
