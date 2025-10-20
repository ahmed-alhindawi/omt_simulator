FROM docker.io/ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential pkg-config tk-dev openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /tmp/build
COPY pyproject.toml uv.lock ./

ARG PYTHON_VERSION=3.13
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} && \
    uv sync --frozen --no-install-project

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /workspace