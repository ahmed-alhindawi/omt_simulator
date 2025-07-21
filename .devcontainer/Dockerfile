FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3.12-venv \
    python3-tk \
    git \
    screen \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/root/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/ pip install -r requirements.txt